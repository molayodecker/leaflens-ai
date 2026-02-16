# LeafLens

Offline-first Android app that scans plant leaves using the device camera and identifies crop diseases. Built with Jetpack Compose (Material 3) and CameraX.

## Features

- **Live camera preview** — back-camera feed via CameraX `PreviewView`
- **Leaf scanning** — capture a frame and classify it with a single tap
- **Disease detection** — identifies conditions across cocoa, maize, and tomato crops
- **Offline guidance** — static agronomic advice for each detected disease
- **No internet required** — runs entirely on-device

## Supported Labels

| Crop   | Disease              |
|--------|----------------------|
| Cocoa  | Black Pod, Healthy   |
| Maize  | Rust, Healthy        |
| Tomato | Early Blight, Healthy|

## Tech Stack

- **Language:** Kotlin
- **UI:** Jetpack Compose + Material 3
- **Camera:** CameraX (camera-core, camera2, lifecycle, view)
- **Async:** Kotlin Coroutines
- **Min SDK:** 24 &nbsp;|&nbsp; **Target SDK:** 35
- **Build:** Gradle Kotlin DSL

## Project Structure

```
app/src/main/java/ai/leaflens/app/
├── MainActivity.kt              # Compose UI + CameraX integration
├── classifier/
│   ├── Classifier.kt            # Interface + Prediction data class
│   └── MockClassifier.kt        # Returns random realistic results
├── ui/
│   └── Theme.kt                 # Material 3 color scheme
└── util/
    └── ImageProxyExt.kt         # YUV_420_888 → Bitmap conversion
```

## Architecture

The `Classifier` interface defines a single method:

```kotlin
interface Classifier {
    suspend fun predict(bitmap: Bitmap?): Prediction
}
```

`MockClassifier` is the current implementation, returning random predictions with simulated latency. To integrate a real model, implement `Classifier` with a `TFLiteClassifier` and swap it in `MainActivity`.

## Build & Run

1. Open the project in Android Studio
2. Sync Gradle
3. Run on a physical device (camera required)

## Permissions

The app requests camera access at launch. If denied, a screen is shown with a button to open system settings.
