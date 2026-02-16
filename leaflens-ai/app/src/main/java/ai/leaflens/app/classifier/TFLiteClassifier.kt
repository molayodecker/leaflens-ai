package ai.leaflens.app.classifier

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteClassifier(context: Context) : Classifier, Closeable {

    private val interpreter: Interpreter
    private val labels: List<String>

    private val inputSize = 224
    private val pixelSize = 3
    private val floatSize = 4

    init {
        val model = loadModelFile(context)
        interpreter = Interpreter(model)
        labels = context.assets.open("labels.txt").bufferedReader().readLines()
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd("model.tflite")
        val inputStream = fd.createInputStream()
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    override suspend fun predict(bitmap: Bitmap?): Prediction = withContext(Dispatchers.Default) {
        if (bitmap == null) {
            return@withContext Prediction(crop = "Unknown", disease = "unknown", confidence = 0f)
        }

        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = preprocessBitmap(resized)

        val outputArray = Array(1) { FloatArray(labels.size) }
        interpreter.run(inputBuffer, outputArray)

        val scores = outputArray[0]
        val maxIdx = scores.indices.maxBy { scores[it] }
        val confidence = scores[maxIdx]

        val label = labels[maxIdx]
        val parts = label.split(":", limit = 2)
        val crop = parts[0]
        val disease = if (parts.size > 1) parts[1] else label

        Prediction(crop = crop, disease = disease, confidence = confidence)
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * pixelSize * floatSize)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            buffer.putFloat((pixel and 0xFF) / 255.0f)           // B
        }

        buffer.rewind()
        return buffer
    }

    override fun close() {
        interpreter.close()
    }
}
