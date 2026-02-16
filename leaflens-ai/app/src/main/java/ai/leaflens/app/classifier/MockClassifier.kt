package ai.leaflens.app.classifier

import android.graphics.Bitmap
import kotlinx.coroutines.delay

class MockClassifier : Classifier {

    private val results = listOf(
        Prediction(crop = "Cocoa", disease = "cocoa_black_pod", confidence = 0.91f),
        Prediction(crop = "Cocoa", disease = "healthy", confidence = 0.97f),
        Prediction(crop = "Maize", disease = "maize_rust", confidence = 0.85f),
        Prediction(crop = "Maize", disease = "healthy", confidence = 0.94f),
        Prediction(crop = "Tomato", disease = "tomato_early_blight", confidence = 0.88f),
        Prediction(crop = "Tomato", disease = "healthy", confidence = 0.96f),
    )

    override suspend fun predict(bitmap: Bitmap?): Prediction {
        // Simulate inference latency
        delay(300)
        return results.random()
    }
}
