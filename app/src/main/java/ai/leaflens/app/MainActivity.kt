package ai.leaflens.app

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Eco
import androidx.compose.material3.Badge
import androidx.compose.material3.BottomSheetScaffold
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SheetValue
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberBottomSheetScaffoldState
import androidx.compose.material3.rememberStandardBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import ai.leaflens.app.classifier.Classifier
import ai.leaflens.app.classifier.MockClassifier
import ai.leaflens.app.classifier.Prediction
import ai.leaflens.app.classifier.TFLiteClassifier
import ai.leaflens.app.ui.LeafLensTheme
import ai.leaflens.app.util.toBitmap
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var classifier: Classifier
    private var isLiveClassifier = false

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)

        classifier = try {
            TFLiteClassifier(this).also {
                Log.i("LeafLens", "Using TFLiteClassifier")
                isLiveClassifier = true
            }
        } catch (e: Exception) {
            Log.w("LeafLens", "TFLite model not found, falling back to MockClassifier", e)
            MockClassifier()
        }

        setContent {
            LeafLensTheme {
                LeafLensApp(classifier, isLiveClassifier)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        (classifier as? TFLiteClassifier)?.close()
    }
}

// ---------------------------------------------------------------------------
// Root composable – handles permission state
// ---------------------------------------------------------------------------

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LeafLensApp(classifier: Classifier, isLive: Boolean) {
    var cameraPermissionGranted by remember { mutableStateOf(false) }
    var permissionRequested by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        cameraPermissionGranted = granted
        permissionRequested = true
    }

    // Request on first composition
    LaunchedEffect(Unit) {
        launcher.launch(Manifest.permission.CAMERA)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Eco,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(24.dp),
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("LeafLens", fontWeight = FontWeight.Bold)
                    }
                },
                actions = {
                    Badge(
                        containerColor = if (isLive)
                            MaterialTheme.colorScheme.primary
                        else
                            MaterialTheme.colorScheme.error,
                        contentColor = if (isLive)
                            MaterialTheme.colorScheme.onPrimary
                        else
                            MaterialTheme.colorScheme.onError,
                    ) {
                        Text(
                            text = if (isLive) "Live" else "Mock",
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                    Spacer(modifier = Modifier.width(12.dp))
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                ),
            )
        },
    ) { innerPadding ->
        Box(modifier = Modifier.padding(innerPadding)) {
            if (cameraPermissionGranted) {
                CameraScreen(classifier)
            } else if (permissionRequested) {
                PermissionDeniedScreen()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Camera screen with preview + scan + bottom sheet results
// ---------------------------------------------------------------------------

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CameraScreen(classifier: Classifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    val imageCapture = remember { ImageCapture.Builder().build() }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    var prediction by remember { mutableStateOf<Prediction?>(null) }
    var scanning by remember { mutableStateOf(false) }

    val sheetState = rememberStandardBottomSheetState(
        initialValue = SheetValue.Hidden,
        skipHiddenState = false,
    )
    val scaffoldState = rememberBottomSheetScaffoldState(bottomSheetState = sheetState)

    // Show/hide the bottom sheet when prediction changes
    LaunchedEffect(prediction) {
        if (prediction != null) {
            sheetState.expand()
        } else {
            sheetState.hide()
        }
    }

    DisposableEffect(Unit) {
        onDispose { cameraExecutor.shutdown() }
    }

    BottomSheetScaffold(
        scaffoldState = scaffoldState,
        sheetPeekHeight = 0.dp,
        sheetContainerColor = MaterialTheme.colorScheme.surface,
        sheetShape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
        sheetShadowElevation = 8.dp,
        sheetContent = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .windowInsetsPadding(WindowInsets.navigationBars)
                    .padding(horizontal = 20.dp, vertical = 8.dp)
                    .animateContentSize(
                        animationSpec = spring(
                            dampingRatio = Spring.DampingRatioMediumBouncy,
                            stiffness = Spring.StiffnessLow,
                        )
                    ),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                prediction?.let { pred ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = "Scan Results",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                        )
                        IconButton(onClick = { prediction = null }) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = "Dismiss",
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    ResultCard(pred)
                    Spacer(modifier = Modifier.height(12.dp))
                    GuidanceCard(pred.label)
                    Spacer(modifier = Modifier.height(16.dp))

                    OutlinedButton(
                        onClick = { prediction = null },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                    ) {
                        Text("Scan Again")
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                }
            }
        },
    ) {
        // Camera preview fills the entire screen
        Box(modifier = Modifier.fillMaxSize()) {
            // Camera preview
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(0.dp))
            ) {
                AndroidView(
                    modifier = Modifier.fillMaxSize(),
                    factory = { ctx ->
                        val previewView = PreviewView(ctx).apply {
                            scaleType = PreviewView.ScaleType.FILL_CENTER
                        }
                        val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                        cameraProviderFuture.addListener({
                            val cameraProvider = cameraProviderFuture.get()
                            val preview = Preview.Builder().build().also {
                                it.setSurfaceProvider(previewView.surfaceProvider)
                            }
                            try {
                                cameraProvider.unbindAll()
                                cameraProvider.bindToLifecycle(
                                    lifecycleOwner,
                                    CameraSelector.DEFAULT_BACK_CAMERA,
                                    preview,
                                    imageCapture,
                                )
                            } catch (e: Exception) {
                                Log.e("LeafLens", "Camera bind failed", e)
                            }
                        }, ContextCompat.getMainExecutor(ctx))
                        previewView
                    },
                )

                // Gradient overlay at the bottom for contrast
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(160.dp)
                        .align(Alignment.BottomCenter)
                        .background(
                            brush = Brush.verticalGradient(
                                colors = listOf(
                                    Color.Transparent,
                                    Color.Black.copy(alpha = 0.5f),
                                )
                            )
                        )
                )
            }

            // Scan button floating at bottom
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .align(Alignment.BottomCenter)
                    .windowInsetsPadding(WindowInsets.navigationBars)
                    .padding(bottom = 32.dp),
                contentAlignment = Alignment.Center,
            ) {
                Button(
                    onClick = {
                        if (scanning) return@Button
                        scanning = true
                        prediction = null

                        imageCapture.takePicture(
                            cameraExecutor,
                            object : ImageCapture.OnImageCapturedCallback() {
                                override fun onCaptureSuccess(image: ImageProxy) {
                                    val bitmap: Bitmap? = image.toBitmap()
                                    image.close()
                                    scope.launch {
                                        prediction = classifier.predict(bitmap)
                                        scanning = false
                                    }
                                }

                                override fun onError(exc: ImageCaptureException) {
                                    Log.e("LeafLens", "Capture failed", exc)
                                    scanning = false
                                }
                            },
                        )
                    },
                    enabled = !scanning,
                    modifier = Modifier.height(56.dp),
                    shape = CircleShape,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                    ),
                    elevation = ButtonDefaults.buttonElevation(
                        defaultElevation = 6.dp,
                    ),
                ) {
                    if (scanning) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.dp,
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Default.CameraAlt,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp),
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Scan Leaf", style = MaterialTheme.typography.titleMedium)
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Result card
// ---------------------------------------------------------------------------

@Composable
private fun ResultCard(prediction: Prediction) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
        ),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Prediction",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
            )
            Spacer(modifier = Modifier.height(8.dp))
            DetailRow("Crop", prediction.crop)
            DetailRow("Disease", prediction.disease.replace('_', ' ').replaceFirstChar { it.uppercase() })
            DetailRow("Confidence", "${prediction.confidencePercent}%")
        }
    }
}

@Composable
private fun DetailRow(label: String, value: String) {
    Column(modifier = Modifier.padding(vertical = 2.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyLarge,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onPrimaryContainer,
        )
    }
}

// ---------------------------------------------------------------------------
// Offline guidance card
// ---------------------------------------------------------------------------

@Composable
private fun GuidanceCard(label: String) {
    val advice = offlineAdvice(label)

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Offline Guidance",
                style = MaterialTheme.typography.labelLarge,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = advice,
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

private fun offlineAdvice(label: String): String = when (label) {
    "healthy" ->
        "The leaf appears healthy. Continue regular watering, " +
                "balanced fertilization, and periodic monitoring for early signs of disease."

    "cocoa_black_pod" ->
        "Black pod disease detected. Remove and destroy infected pods immediately. " +
                "Improve canopy airflow through pruning. Apply a copper-based fungicide " +
                "during the rainy season as a preventive measure."

    "maize_rust" ->
        "Maize rust detected. Remove severely affected leaves. " +
                "Consider applying a foliar fungicide (e.g., triazole-based). " +
                "For future plantings, choose rust-resistant maize varieties."

    "tomato_early_blight" ->
        "Early blight detected on tomato. Remove infected lower leaves. " +
                "Avoid overhead irrigation. Apply a fungicide containing chlorothalonil " +
                "or copper hydroxide. Rotate crops to reduce soil-borne inoculum."

    else ->
        "No specific guidance available for this label. " +
                "Consult a local agricultural extension officer for personalized advice."
}

// ---------------------------------------------------------------------------
// Permission-denied screen
// ---------------------------------------------------------------------------

@Composable
private fun PermissionDeniedScreen() {
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Camera Permission Required",
            style = MaterialTheme.typography.headlineSmall,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "LeafLens needs camera access to scan leaves and identify diseases. " +
                    "Please grant the permission in Settings.",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = {
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                    data = Uri.fromParts("package", context.packageName, null)
                }
                context.startActivity(intent)
            },
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary,
            ),
            shape = RoundedCornerShape(12.dp),
        ) {
            Text("Open Settings")
        }
    }
}
