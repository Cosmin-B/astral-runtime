// AstralEditorUtilities.cs - Unity Editor utilities for Astral runtime
//
// Provides validation, diagnostics, and editor-time helpers

#if UNITY_EDITOR
using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEditor;

namespace Astral.Runtime.Editor
{
    /// <summary>
    /// Unity Editor utilities for Astral runtime.
    /// Provides validation, diagnostics, and editor-time helpers.
    /// </summary>
    public static class AstralEditorUtilities
    {
        [MenuItem("Tools/Astral/Validate Native Plugins")]
        public static void ValidateNativePlugins()
        {
            Debug.Log("[Astral] Validating native plugins...");

            bool allValid = true;

            // Check platform-specific plugins
            allValid &= ValidatePlugin("Runtime/Plugins/x86_64/libastral_rt.so", "Linux x86_64");
            allValid &= ValidatePlugin("Runtime/Plugins/x86_64/astral_rt.dll", "Windows x86_64");
            allValid &= ValidatePlugin("Runtime/Plugins/arm64/libastral_rt.so", "Android ARM64");
            allValid &= ValidatePlugin("Runtime/Plugins/arm64/libastral_rt.dylib", "macOS ARM64");
            allValid &= ValidatePlugin("Runtime/Plugins/iOS/libastral_rt.a", "iOS ARM64");

            if (allValid)
            {
                Debug.Log("[Astral] All native plugins validated successfully");
            }
            else
            {
                Debug.LogWarning("[Astral] Some native plugins are missing. See Runtime/Plugins/README.md for build instructions.");
            }
        }

        [MenuItem("Tools/Astral/Test Runtime Initialization")]
        public static void TestRuntimeInitialization()
        {
            Debug.Log("[Astral] Testing runtime initialization...");

            try
            {
                if (AstralRuntime.IsInitialized)
                {
                    Debug.LogWarning("[Astral] Runtime already initialized. Shutting down first...");
                    AstralRuntime.Shutdown();
                }

                // Initialize with default config
                AstralRuntime.Initialize(AstralConfig.Default);

                if (AstralRuntime.IsInitialized)
                {
                    Debug.Log("[Astral] Runtime initialized successfully");

                    // Test error string function
                    string errorMsg = AstralRuntime.GetErrorString(AstralNative.ASTRAL_E_INVALID);
                    Debug.Log($"[Astral] Error string test: ASTRAL_E_INVALID = '{errorMsg}'");

                    // Shutdown
                    AstralRuntime.Shutdown();
                    Debug.Log("[Astral] Runtime shutdown successfully");
                }
                else
                {
                    Debug.LogError("[Astral] Runtime initialization failed");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Astral] Runtime initialization test failed: {ex.Message}");
            }
        }

        [MenuItem("Tools/Astral/Show Runtime Info")]
        public static void ShowRuntimeInfo()
        {
            Debug.Log("[Astral] Runtime Information:");
            Debug.Log($"  Initialized: {AstralRuntime.IsInitialized}");

            if (AstralRuntime.IsInitialized)
            {
                var config = AstralRuntime.Config;
                Debug.Log($"  Reserve Bytes: {config.reserveBytes / (1024 * 1024)} MB");
                Debug.Log($"  Thread Count: {config.threadCount} (0 = auto-detect)");
                Debug.Log($"  Huge Pages: {config.enableHugePages}");
            }

            // Platform info
            Debug.Log($"  Platform: {Application.platform}");
            Debug.Log($"  Unity Version: {Application.unityVersion}");
            Debug.Log($"  Scripting Backend: {GetScriptingBackend()}");
            Debug.Log($"  IL2CPP: {IsIL2CPP()}");
        }

        [MenuItem("Tools/Astral/Open Documentation")]
        public static void OpenDocumentation()
        {
            Application.OpenURL("https://github.com/astral-runtime/astral");
        }

        private static bool ValidatePlugin(string relativePath, string platformName)
        {
            // Find Astral package directory
            string packagePath = FindPackagePath();
            if (string.IsNullOrEmpty(packagePath))
            {
                Debug.LogError("[Astral] Could not find Astral package directory");
                return false;
            }

            string fullPath = Path.Combine(packagePath, relativePath);
            bool exists = File.Exists(fullPath);

            if (exists)
            {
                FileInfo fileInfo = new FileInfo(fullPath);
                Debug.Log($"[Astral] {platformName}: OK ({fileInfo.Length / 1024} KB)");
            }
            else
            {
                Debug.LogWarning($"[Astral] {platformName}: MISSING ({fullPath})");
            }

            return exists;
        }

        private static string FindPackagePath()
        {
            // Search for package in Packages/ directory
            string[] packagePaths = new string[]
            {
                "Packages/com.astral.runtime",
                "Assets/Astral.Runtime",
                Path.Combine(Application.dataPath, "..", "Packages", "com.astral.runtime")
            };

            foreach (string path in packagePaths)
            {
                if (Directory.Exists(path))
                {
                    return path;
                }
            }

            return null;
        }

        private static string GetScriptingBackend()
        {
            var backend = PlayerSettings.GetScriptingBackend(EditorUserBuildSettings.selectedBuildTargetGroup);
            return backend.ToString();
        }

        private static bool IsIL2CPP()
        {
            var backend = PlayerSettings.GetScriptingBackend(EditorUserBuildSettings.selectedBuildTargetGroup);
            return backend == ScriptingImplementation.IL2CPP;
        }
    }

    /// <summary>
    /// Custom inspector for AstralRuntimeInitializer component.
    /// </summary>
    [CustomEditor(typeof(AstralRuntimeInitializer))]
    public class AstralRuntimeInitializerEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Runtime Status", EditorStyles.boldLabel);

            if (Application.isPlaying)
            {
                EditorGUILayout.LabelField("Initialized", AstralRuntime.IsInitialized ? "Yes" : "No");

                if (AstralRuntime.IsInitialized)
                {
                    var config = AstralRuntime.Config;
                    EditorGUILayout.LabelField("Reserve Memory", $"{config.reserveBytes / (1024 * 1024)} MB");
                    EditorGUILayout.LabelField("Thread Count", config.threadCount == 0 ? "Auto" : config.threadCount.ToString());
                    EditorGUILayout.LabelField("Huge Pages", config.enableHugePages ? "Enabled" : "Disabled");
                }
            }
            else
            {
                EditorGUILayout.HelpBox("Enter Play mode to see runtime status", MessageType.Info);
            }
        }
    }

    /// <summary>
    /// Custom property drawer for AstralConfig.
    /// </summary>
    [CustomPropertyDrawer(typeof(AstralConfig))]
    public class AstralConfigDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            EditorGUI.BeginProperty(position, label, property);

            var reserveBytesProperty = property.FindPropertyRelative("reserveBytes");
            var threadCountProperty = property.FindPropertyRelative("threadCount");
            var hugePagesProp = property.FindPropertyRelative("enableHugePages");

            float lineHeight = EditorGUIUtility.singleLineHeight;
            float spacing = EditorGUIUtility.standardVerticalSpacing;

            Rect rect = new Rect(position.x, position.y, position.width, lineHeight);

            EditorGUI.LabelField(rect, label, EditorStyles.boldLabel);
            rect.y += lineHeight + spacing;

            EditorGUI.indentLevel++;

            // Reserve bytes (convert to MB for readability)
            ulong reserveBytes = (ulong)reserveBytesProperty.longValue;
            int reserveMB = (int)(reserveBytes / (1024 * 1024));
            int newReserveMB = EditorGUI.IntField(rect, "Reserve (MB)", reserveMB);
            if (newReserveMB != reserveMB)
            {
                reserveBytesProperty.longValue = (long)((ulong)newReserveMB * 1024 * 1024);
            }
            rect.y += lineHeight + spacing;

            // Thread count
            EditorGUI.PropertyField(rect, threadCountProperty, new GUIContent("Thread Count (0 = auto)"));
            rect.y += lineHeight + spacing;

            // Huge pages
            EditorGUI.PropertyField(rect, hugePagesProp, new GUIContent("Enable Huge Pages"));

            EditorGUI.indentLevel--;

            EditorGUI.EndProperty();
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return EditorGUIUtility.singleLineHeight * 4 + EditorGUIUtility.standardVerticalSpacing * 3;
        }
    }
}
#endif
