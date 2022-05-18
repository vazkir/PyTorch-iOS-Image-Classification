//
//  Predictor.swift
//  PyTorchDemo
//
//  Created by Vasco Meerman on 18/05/2022.
//
// This is the interface which loads our model and communicates with PyTorch


import Foundation

struct InferenceResult {
    let score: Float
    let label: String
    
    func scoreAndResultPretty() -> String {
        let scorePercentageStr = String(format:"%.2f", score)
        return "\(scorePercentageStr)% -> \(label)"
    }
}


class Predictor {
    private var isRunning: Bool = false

    
    private lazy var module: VisionTorchModule = {
        // Create a filePath object by specifiying the path in our app bundle
        // And try to conditionally open this module at the path, else raise an error
        if let filePath = Bundle.main.path(forResource: "mobilenet_quantized", ofType: "pt"),
           let module = VisionTorchModule(fileAtPath: filePath) {
            return module
        }
        fatalError("Failed to load model.....")
    }()
        
    private lazy var labels: [String] = {
        // Create a filePath object by specifiying the path in our app bundle
        // And try to conditionally open this module at the path, else raise an error
        if let filePath = Bundle.main.path(forResource: "labels", ofType: "txt"),
           let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        }
        fatalError("Failed to load labels file.....")
    }()
    
    
    func predict(_ buffer: [Float32], resultCount: Int) -> [InferenceResult]? {
        // This check needs to happen, because is if the buffer keeps pushing new images
        // The predictors will be overwhelmed
        if isRunning{return nil}
        
        isRunning = true
        
        var tensorBuffer = buffer
        
        // Now we try to predict on our actual loaded TorchScript model by supplying our image buffer float array
        // UnsafeMutablePointer can be used type to access data of a specific type in memory. The input given is what it points to
        // It provides no automated memory management or alignment guarantees -> You responsible for life cicle
        guard let outputs = module.predict(image: UnsafeMutablePointer(&tensorBuffer)) else{
            return nil
        }
        
        isRunning = false
        
        return topK(scores: outputs, labels: labels, count: resultCount)
    }
    
    
    // Takes in the results from the model and combines it with the labels and
    // returns all the results in an array
    func topK(scores: [NSNumber], labels: [String], count: Int) -> [InferenceResult] {
        // Merge the labels and scoress together so we can start sorting them together
        let zippedResults = zip(labels.indices, scores)
        
        // Sorts the labels with scores based on the highest floatValues aka accuracy in this case
        let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue}.prefix(count)
        
        // This maps the sorted array to the struct object in an array
        return sortedResults.map { InferenceResult(score: $0.1.floatValue, label: labels[$0.0]) }
    }
}
