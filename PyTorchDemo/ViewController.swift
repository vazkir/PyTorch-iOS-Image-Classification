//
//  ViewController.swift
//  PyTorchDemo
//
//  Created by Vasco Meerman on 18/05/2022.
//

import UIKit
import AVFoundation

class ViewController: UIViewController {

    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var previewView: UIView!
    
    private var captureSession: AVCaptureSession!
    private var videoPreviewLayer: AVCaptureVideoPreviewLayer!
    private var videoOutput = AVCaptureVideoDataOutput() // The buffer we can feed to the model
    
    // Threads needed for concurrency reasons, so all the processes can run simultanously
    private var sessionQueue = DispatchQueue(label: "session")
    private var bufferQueue = DispatchQueue(label: "buffer")
    
    // Our custom class for predictions
    private var predictor = Predictor()

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        startSession()
    }
    
    private func startSession() {
        captureSession = AVCaptureSession()
        
        sessionQueue.async {
            self.captureSession.sessionPreset = .high
            
            // We define that our config setup started for the capture session
            // which manages the input from the camera and delegates the output
            self.captureSession.beginConfiguration()
            self.configureCamera()
            self.configureOutput()
            self.captureSession.commitConfiguration() // Make sure to commit after we're done
            
            // This prepare method makes sure we get the permissions from the user to use the camera
            self.prepare {
                if $0, !self.captureSession.isRunning {
                    self.captureSession.startRunning()
                }
            }
        }
    }
    
    // This creates a UI reference to our preview
    private func setupLivePreview() {
        // We bind the capture session which opens and gets input from the camera
        // to the video preview layer to show on the UI
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        
        // How the video will be showsn, aspect fit (I think?)
        videoPreviewLayer.videoGravity = .resizeAspect
        
        // Make sure that the video stays in portrait mode
        videoPreviewLayer.connection?.videoOrientation = .portrait
        
        // Now that the setup has been done, we can add it to our UI View
        previewView.layer.addSublayer(videoPreviewLayer)

        // Dispath the live preview to a high non blocking thread
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
            
            // UI Updates need to happen on the main thead only!
            DispatchQueue.main.async {
                // Make sure the videoPreview keeps the bounds of the view on which it's projects
                self.videoPreviewLayer.frame = self.previewView.bounds
            }
        }
    }
    
    private func getBackcamera() -> AVCaptureDevice? {
        // We grab the wideangle camaera for video recording of the back of the device
        if let device = AVCaptureDevice.default(.builtInDualCamera,
                                                for: .video, position: .back) {
            return device
        } else if let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                       for: .video, position: .back) {
            return device
        } else {
            fatalError("Missing expected back camera device.")
        }
    }
    
    func configureCamera() {
        guard let camera = getBackcamera() else {
            return
        }
        
        // We create the input, so we can capture the content of the actual camera
        let input = try! AVCaptureDeviceInput(device: camera)
        
        if captureSession.canAddInput(input){
            captureSession.addInput(input)
        }
    }

    
    func configureOutput(){
        // Delegate the video output buffer to it's own que
        // See captureOutput for the output of this buffer
        videoOutput.setSampleBufferDelegate(self, queue: bufferQueue)
        
        // Video frames coming in too late, will not be used (I think)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        
        // We set the output type of the video buffer, which is 32BGRA (not 100% sure)
        videoOutput.videoSettings = [String(kCVPixelBufferPixelFormatTypeKey): kCMPixelFormat_32BGRA]
        
        // We are connecting our capture session where put all the pixels in and map them to our videoOutput
        guard captureSession.canAddOutput(videoOutput) else {
            return
        }
        
        // We add the actual buffered video output to our capture session
        captureSession.addOutput(videoOutput)
        
        DispatchQueue.main.async {
            self.setupLivePreview()
        }
    }
    
    // Checks if access to camera has already been granted, and if not will trigger native popup
    // You can see our required part of the native popup text in the info.plist
    private func prepare(_ completionHandler: @escaping (Bool) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        if status == .notDetermined {
            AVCaptureDevice.requestAccess(for: .video) {
                completionHandler($0)
            }
            return
        }
        completionHandler(status == .authorized)
    }
    
    func process(buffer: [Float]?){
        // Unwrap the optional buffer to make sure there is no nul value
        guard let pixelBuffer = buffer else { return }
        
        // Also speficy the amount of results we want to get back
        let result = predictor.predict(pixelBuffer, resultCount: 2)
        
        // UI Updates need to happen on the main thead only!
        DispatchQueue.main.async {
            // Display the result on the frontend
            self.resultLabel.text = result?.first?.scoreAndResultPretty()
        }
    }
}

// Required for AVCaptureVideoDataOutput setSampleBufferDelegate delegation of the output buffer
extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Make sure it's portrait
        connection.videoOrientation = .portrait
        
        // First we have to retrieve our pixel buffer
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // Now we can use our normalization function to get the output in the right format for the the model as input
        guard let normalizedBuffer = pixelBuffer.normalized(224, 224) else {
            return
        }
        
        // Now the formatted output buffer is ready to be processed by our TorchScript model
        process(buffer: normalizedBuffer)
    }
}


