//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
@testable import WhisperKit
import XCTest

/// Top-k sampling when the cumulative top-k probabilities tie (float32 saturation at 1.0).
final class TokenSamplerTieTests: XCTestCase {
    private let endOfText = 50257
    private let transcribe = 50360
    private let korean = 50264

    // Top-5 probabilities of the logits below at temperature 0.2: every cumulative sum rounds to 1.0 in float32.
    private let saturatedProbs: [Float] = [1.0, 1.9e-25, 5.8e-30, 7.4e-31, 6.1e-32]
    private let saturatedIds: [Int32] = [50360, 50264, 50257, 50259, 50261]

    private func requireMLTensor() throws {
        guard #available(macOS 15, iOS 18, watchOS 11, visionOS 2, *) else {
            throw XCTSkip("MLTensor sampling path needs macOS 15 / iOS 18")
        }
    }

    @available(macOS 15, iOS 18, watchOS 11, visionOS 2, *)
    private func tensor<Scalar: MLTensorScalar>(_ values: [Scalar]) -> MLTensor {
        MLTensor(shape: [1, 1, values.count], scalars: values)
    }

    /// The selection used before the tie fix, kept here to compare untied draws.
    @available(macOS 15, iOS 18, watchOS 11, visionOS 2, *)
    private func argsortSelection(probs: [Float], ids: [Int32], uniform: Float) -> Int {
        let topKProbs = tensor(probs)
        let topKIndices = tensor(ids)
        let rnd = topKProbs.sum() * uniform
        var accumTopKProbs = topKProbs.cumulativeSum(alongAxis: -1)
        accumTopKProbs += (accumTopKProbs .< rnd) * 100.0
        let topKIndex = accumTopKProbs.argsort()[..., 0]
        return topKIndices.gathering(atIndices: topKIndex, alongAxis: topKIndices.rank - 1).asIntArray()[0]
    }

    /// Logits of shape [1, 1, vocab] filled with `fill`, with the given token logits set.
    private func makeLogits(_ values: [Int: Float], vocab: Int = 51866, fill: Float = -10) throws -> MLMultiArray {
        let dataType: MLMultiArrayDataType = FloatType.self == Float16.self ? .float16 : .float32
        let logits = try MLMultiArray(shape: [1, 1, NSNumber(value: vocab)], dataType: dataType)
        let pointer = logits.dataPointer.bindMemory(to: FloatType.self, capacity: vocab)
        for index in 0..<vocab {
            pointer[index] = FloatType(values[index] ?? fill)
        }
        return logits
    }

    // ① Saturated cumulative sums pick position 0 for every random draw.
    func testSaturatedCumulativeSumPicksMostProbableForAnyDraw() throws {
        try requireMLTensor()
        guard #available(macOS 15, iOS 18, watchOS 11, visionOS 2, *) else { return }
        for step in 0..<1000 {
            let uniform = Float(step) / 1000
            let result = GreedyTokenSampler.sampleTopK(
                topKProbs: tensor(saturatedProbs), topKIndices: tensor(saturatedIds), uniform: uniform
            )
            XCTAssertEqual(result.token, transcribe, "uniform=\(uniform)")
            XCTAssertEqual(result.logprob, 0, accuracy: 1e-6)
        }
    }

    // ① Through the sampler: fallback-temperature logits in the saturating pattern never sample the end-of-text token.
    func testFallbackTemperatureDoesNotSampleNearZeroEndOfText() throws {
        try requireMLTensor()
        let logits = try makeLogits([transcribe: 18.88, korean: 7.49, endOfText: 5.41, 50259: 5.0, 50261: 4.5])
        let sampler = GreedyTokenSampler(temperature: 0.2, eotToken: endOfText, decodingOptions: DecodingOptions())
        for _ in 0..<50 {
            let result = sampler.update(tokens: [50258, korean], logits: logits, logProbs: [0, 0])
            XCTAssertEqual(result.tokens.last, transcribe)
            XCTAssertFalse(result.completed)
        }
    }

    // ② Untied distributions select the same position as before, so the sampling distribution is unchanged.
    func testUntiedDistributionMatchesPreviousSelection() throws {
        try requireMLTensor()
        guard #available(macOS 15, iOS 18, watchOS 11, visionOS 2, *) else { return }
        let probs: [Float] = [0.4, 0.3, 0.2, 0.07, 0.03]
        let ids: [Int32] = [11, 22, 33, 44, 55]
        let draws = 1000
        var counts = [Int: Int]()
        for step in 0..<draws {
            let uniform = (Float(step) + 0.5) / Float(draws)
            let result = GreedyTokenSampler.sampleTopK(topKProbs: tensor(probs), topKIndices: tensor(ids), uniform: uniform)
            XCTAssertEqual(result.token, argsortSelection(probs: probs, ids: ids, uniform: uniform), "uniform=\(uniform)")
            XCTAssertEqual(result.logprob, log(probs[ids.firstIndex(of: Int32(result.token))!]), accuracy: 1e-6)
            counts[result.token, default: 0] += 1
        }
        for (prob, id) in zip(probs, ids) {
            XCTAssertEqual(Double(counts[Int(id)] ?? 0) / Double(draws), Double(prob), accuracy: 0.002)
        }
    }

    func testFirstMinimumPositionResolvesTiesToEarliest() {
        XCTAssertEqual(GreedyTokenSampler.firstMinimumPosition([1, 1, 1, 1, 1]), 0)
        XCTAssertEqual(GreedyTokenSampler.firstMinimumPosition([100.4, 100.7, 0.9, 1.0, 1.0]), 2)
        XCTAssertEqual(GreedyTokenSampler.firstMinimumPosition([100.4, 1.0, 1.0, 1.0, 1.0]), 1)
    }

    // ④ When the end-of-text token really is the top-1, it is still sampled.
    func testEndOfTextAsTopOneIsStillSampled() throws {
        try requireMLTensor()
        let logits = try makeLogits([endOfText: 18.88, transcribe: 7.49, korean: 5.41, 50259: 5.0, 50261: 4.5])
        let sampler = GreedyTokenSampler(temperature: 0.2, eotToken: endOfText, decodingOptions: DecodingOptions())
        for _ in 0..<50 {
            let result = sampler.update(tokens: [50258, korean], logits: logits, logProbs: [0, 0])
            XCTAssertEqual(result.tokens.last, endOfText)
            XCTAssertTrue(result.completed)
        }
    }

    // ⑤ Temperature 0 (argmax) returns the same token and bit-identical logprob as the tensor ops it has always used.
    func testTemperatureZeroArgmaxIsUnchanged() throws {
        try requireMLTensor()
        guard #available(macOS 15, iOS 18, watchOS 11, visionOS 2, *) else { return }
        let logits = try makeLogits([transcribe: 18.88, korean: 7.49, endOfText: 5.41, 50259: 5.0, 50261: 4.5])
        let sampler = GreedyTokenSampler(temperature: 0, eotToken: endOfText, decodingOptions: DecodingOptions())
        let result = sampler.update(tokens: [50258, korean], logits: logits, logProbs: [0, 0])

        let softmaxScores = MLTensor(MLShapedArray<FloatType>(logits)).cast(to: Float.self).softmax(alongAxis: -1)
        let argmax = softmaxScores.argmax(alongAxis: -1)
        let expectedLogprob = softmaxScores.gathering(atIndices: argmax, alongAxis: -1).log().asFloatArray()[0]
        XCTAssertEqual(result.tokens.last, argmax.asIntArray()[0])
        XCTAssertEqual(result.tokens.last, transcribe)
        XCTAssertEqual(result.logProbs.last?.bitPattern, expectedLogprob.bitPattern)
    }
}
