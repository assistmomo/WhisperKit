//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML
import Foundation

public protocol TokenSampling {
    func update(tokens: [Int], logits: MLMultiArray, logProbs: [Float]) -> SamplingResult
    func finalize(tokens: [Int], logProbs: [Float]) -> SamplingResult
}

public struct SamplingResult: Sendable {
    public var tokens: [Int]
    public var logProbs: [Float]
    public var completed: Bool

    public init(
        tokens: [Int],
        logProbs: [Float],
        completed: Bool
    ) {
        self.tokens = tokens
        self.logProbs = logProbs
        self.completed = completed
    }
}

open class GreedyTokenSampler: TokenSampling {
    public var temperature: FloatType
    public var eotToken: Int
    public var decodingOptions: DecodingOptions

    public init(temperature: FloatType, eotToken: Int, decodingOptions: DecodingOptions) {
        self.temperature = temperature
        self.eotToken = eotToken
        self.decodingOptions = decodingOptions
    }

    #if canImport(CoreML.MLState)
    @available(macOS 15, iOS 18, watchOS 11, visionOS 2, *)
    private func sampleWithMLTensor(logits: MLMultiArray) -> (token: Int, logprob: Float) {
        // Use MLTensor operations if available for sampling
        // Reference: https://github.com/huggingface/swift-transformers/blob/preview/Sources/Generation/Decoders.swift
        var logitsTensor = MLTensor(MLShapedArray<FloatType>(logits)).cast(to: Float.self)
        let nextTokenTensor: MLTensor
        let nextLogprobTensor: MLTensor

        if temperature != 0.0 {
            // Scale logits by temperature if > 0
            logitsTensor = logitsTensor / temperature
        }

        // Always softmax once
        let softmaxScores = logitsTensor.softmax(alongAxis: -1)

        if temperature != 0.0 {
            // top-k multinomial sampling
            let (topKProbs, topKIndices) = softmaxScores.topK(decodingOptions.topK)
            return Self.sampleTopK(topKProbs: topKProbs, topKIndices: topKIndices, uniform: Float.random(in: 0..<1))
        } else {
            nextTokenTensor = logitsTensor.argmax(alongAxis: -1)
            nextLogprobTensor = softmaxScores.gathering(atIndices: nextTokenTensor, alongAxis: -1).log()
        }

        return (
            token: nextTokenTensor.asIntArray()[0],
            logprob: nextLogprobTensor.asFloatArray()[0]
        )
    }

    /// One top-k multinomial draw from `topKProbs` (probabilities in descending order) and their token ids.
    /// `uniform` is the random draw in `[0, 1)`; it is a parameter so tests can pin it.
    ///
    /// The previous selection, `accumTopKProbs.argsort()[..., 0]`, relies on the order `argsort` gives to equal keys,
    /// which is unspecified. When the tail probabilities fall below float32 resolution relative to the top-1
    /// (e.g. `[1.0, 1.9e-25, 5.8e-30, 7.4e-31, 6.1e-32]`), every cumulative sum rounds to `1.0`, no key receives the
    /// `+100` offset, and `argsort` can return a later position (position 2 on macOS 26, Apple Silicon). When that
    /// token is end-of-text, the fallback pass ends with an empty transcription. Equal keys now resolve to the earliest
    /// (most probable) position. Draws without ties pick the same position as before.
    @available(macOS 15, iOS 18, watchOS 11, visionOS 2, *)
    static func sampleTopK(topKProbs: MLTensor, topKIndices: MLTensor, uniform: Float) -> (token: Int, logprob: Float) {
        let rnd = topKProbs.sum() * uniform
        var accumTopKProbs = topKProbs.cumulativeSum(alongAxis: -1)
        accumTopKProbs += (accumTopKProbs .< rnd) * 100.0
        let position = firstMinimumPosition(accumTopKProbs.asFloatArray())
        return (
            token: topKIndices.asIntArray()[position],
            logprob: log(topKProbs.asFloatArray()[position])
        )
    }
    #endif

    /// Position of the smallest key; equal keys resolve to the earliest position.
    static func firstMinimumPosition(_ keys: [Float]) -> Int {
        var best = 0
        for position in keys.indices where keys[position] < keys[best] {
            best = position
        }
        return best
    }

    private func sampleWithBNNS(logits: MLMultiArray) -> (token: Int, logprob: Float) {
        // TODO: BNNS operations here are deprecated, replace with vDSP or MLX
        var softmaxOutput: BNNSNDArrayDescriptor?
        var argmaxOutput: BNNSNDArrayDescriptor?
        var softmaxInput: BNNSNDArrayDescriptor?
        var softmaxInputNeedsDeallocate = false

        var nextToken: Int?

        do {
            let logitsRawPointer = UnsafeMutableRawBufferPointer(
                start: logits.dataPointer,
                count: logits.count * MemoryLayout<FloatType>.stride
            )

            let logitsDescriptor = BNNSNDArrayDescriptor(
                data: logitsRawPointer,
                scalarType: FloatType.self,
                shape: .vector(logits.count, stride: 1)
            )!

            softmaxInput = logitsDescriptor

            // Scale logits by temperature if > 0
            if temperature != 0.0 {
                let scaledLogits = BNNSNDArrayDescriptor.allocateUninitialized(
                    scalarType: FloatType.self,
                    shape: .vector(logits.count, stride: 1)
                )

                try! BNNS.applyActivation(
                    activation: BNNS.ActivationFunction.linear(alpha: Float(1 / temperature)),
                    input: logitsDescriptor,
                    output: scaledLogits,
                    batchSize: 1
                )

                softmaxInput = scaledLogits
                softmaxInputNeedsDeallocate = true
            }

            // Always softmax once
            softmaxOutput = BNNSNDArrayDescriptor.allocateUninitialized(
                scalarType: Float.self,
                shape: .vector(logits.count, stride: 1)
            )

            try BNNS.applyActivation(
                activation: BNNS.ActivationFunction.softmax,
                input: softmaxInput!,
                output: softmaxOutput!,
                batchSize: 1
            )

            if temperature != 0.0 {
                // top-k multinomial sampling
                let k = decodingOptions.topK
                let bestValues = BNNSNDArrayDescriptor.allocateUninitialized(
                    scalarType: Float.self,
                    shape: .vector(k, stride: 1)
                )
                let bestIndices = BNNSNDArrayDescriptor.allocateUninitialized(
                    scalarType: Int32.self,
                    shape: .vector(k, stride: 1)
                )

                try! BNNS.applyTopK(
                    k: k,
                    input: softmaxOutput!,
                    bestValues: bestValues,
                    bestIndices: bestIndices,
                    axis: 0,
                    batchSize: 1
                )

                let bestValuesResult = bestValues.makeArray(of: Float.self)!
                let bestIndicesResult = bestIndices.makeArray(of: Int32.self)!

                bestValues.deallocate()
                bestIndices.deallocate()

                // multinomial sample from top-k
                let sumOfbestIndicesResult = bestValuesResult.reduce(0, +)
                let rnd = Float.random(in: 0..<sumOfbestIndicesResult)
                var accumulator = Float(0.0)
                var chosenIndex = 0
                for i in 0..<bestValuesResult.count {
                    accumulator += bestValuesResult[i]
                    if rnd < accumulator {
                        chosenIndex = i
                        break
                    }
                }

                nextToken = Int(bestIndicesResult[chosenIndex])
            } else {
                argmaxOutput = BNNSNDArrayDescriptor.allocateUninitialized(
                    scalarType: Float.self,
                    shape: .vector(1, stride: 1)
                )

                try! BNNS.applyReduction(
                    BNNS.ReductionFunction.argMax,
                    input: logitsDescriptor,
                    output: argmaxOutput!,
                    weights: nil
                )

                let argmaxResult = argmaxOutput!.makeArray(of: Float.self)!

                nextToken = Int(argmaxResult[0])
            }
        } catch {
            Logging.error("Sampling error: \(error)")
        }

        // Log of softmax probability of chosen token
        let softmaxResult = softmaxOutput!.makeArray(of: Float.self)!
        let nextLogprob = log(Float(softmaxResult[nextToken!]))
        // Deallocations
        softmaxOutput?.deallocate()
        argmaxOutput?.deallocate()
        if softmaxInputNeedsDeallocate {
            softmaxInput?.deallocate()
        }

        return (token: nextToken!, logprob: nextLogprob)
    }

    public func update(tokens: [Int], logits: MLMultiArray, logProbs: [Float]) -> SamplingResult {
        var nextTokens = tokens
        var nextLogprobs = logProbs
        var completed = false

        var result: (token: Int, logprob: Float)
        #if canImport(CoreML.MLState)
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            result = sampleWithMLTensor(logits: logits)
        } else {
            result = sampleWithBNNS(logits: logits)
        }
        #else
        result = sampleWithBNNS(logits: logits)
        #endif

        nextTokens = tokens + [result.token]
        nextLogprobs = logProbs + [result.logprob]
        completed = result.token == eotToken

        return SamplingResult(
            tokens: nextTokens,
            logProbs: nextLogprobs,
            completed: completed
        )
    }

    public func finalize(tokens: [Int], logProbs: [Float]) -> SamplingResult {
        var finalTokens = tokens
        var finalLogProbs = logProbs
        if tokens.last != eotToken {
            finalTokens.append(eotToken)
            finalLogProbs.append(0)
        }

        return SamplingResult(tokens: finalTokens, logProbs: finalLogProbs, completed: true)
    }
}

open class BeamSearchTokenSampler: TokenSampling {
    public var beamSize: Int
    public var eotToken: Int
    public var patience: Float
    var maxCandidates: Int
    var finishedSequences: [Float]

    public init(
        beamSize: Int,
        eotToken: Int,
        patience: Float = 1
    ) {
        self.beamSize = beamSize
        self.eotToken = eotToken
        self.patience = patience
        self.maxCandidates = Int(Float(beamSize) * patience)
        self.finishedSequences = []
        if self.maxCandidates <= 0 {
            self.maxCandidates = 1
            fatalError("Invalid beam size \(beamSize) or patience \(patience)")
        }
    }

    public func reset() {
        finishedSequences = []
    }

    public func update(tokens: [Int], logits: MLMultiArray, logProbs: [Float]) -> SamplingResult {
        // TODO: Implement
        fatalError("Not implemented: \(#function)")
    }

    public func finalize(tokens: [Int], logProbs: [Float]) -> SamplingResult {
        // TODO: Implement
        fatalError("Not implemented: \(#function)")
    }
}
