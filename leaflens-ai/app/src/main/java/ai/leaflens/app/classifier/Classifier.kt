package ai.leaflens.app.classifier

import android.graphics.Bitmap

data class Prediction(
    val crop: String,
    val disease: String,
    val confidence: Float,
) {
    val label: String
        get() = if (disease == "healthy") "healthy" else disease

    val confidencePercent: Int
        get() = (confidence * 100).toInt()
}

interface Classifier {
    suspend fun predict(bitmap: Bitmap?): Prediction
}
