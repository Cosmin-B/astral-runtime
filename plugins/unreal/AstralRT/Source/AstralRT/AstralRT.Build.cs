using UnrealBuildTool;
using System.IO;

public class AstralRT : ModuleRules
{
    private static string RequireThirdPartyFile(string Path, string Label)
    {
        if (!File.Exists(Path))
        {
            throw new BuildException(
                "AstralRT {0} is missing: {1}. Run: cmake --preset unreal-plugin && cmake --build --preset unreal-plugin -j",
                Label,
                Path
            );
        }

        return Path;
    }

    public AstralRT(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine"
        });

        // ThirdParty: AstralCore
        string ThirdPartyPath = Path.Combine(ModuleDirectory, "../ThirdParty/AstralCore");
        string IncludePath = Path.Combine(ThirdPartyPath, "include");
        string LibPath = Path.Combine(ThirdPartyPath, "lib");

        RequireThirdPartyFile(Path.Combine(IncludePath, "astral_rt.h"), "C ABI header");
        PublicIncludePaths.Add(IncludePath);

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string StaticLib = Path.Combine(LibPath, "Win64", "astral_rt.lib");
            string StaticLibAlt = Path.Combine(LibPath, "Win64", "astral_rt_static.lib");
            PublicAdditionalLibraries.Add(RequireThirdPartyFile(
                File.Exists(StaticLib) ? StaticLib : StaticLibAlt,
                "Win64 static library"
            ));
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            string LinuxLibPath = Path.Combine(LibPath, "Linux");
            string LinuxArchive = RequireThirdPartyFile(Path.Combine(LinuxLibPath, "libastral_rt.a"), "Linux static library");
            RequireThirdPartyFile(Path.Combine(LinuxLibPath, "libllama.a"), "Linux llama static library");
            RequireThirdPartyFile(Path.Combine(LinuxLibPath, "libllama-common.a"), "Linux llama-common static library");
            RequireThirdPartyFile(Path.Combine(LinuxLibPath, "libllama-common-base.a"), "Linux llama-common-base static library");
            RequireThirdPartyFile(Path.Combine(LinuxLibPath, "libggml.a"), "Linux ggml static library");
            RequireThirdPartyFile(Path.Combine(LinuxLibPath, "libggml-cpu.a"), "Linux ggml-cpu static library");
            RequireThirdPartyFile(Path.Combine(LinuxLibPath, "libggml-base.a"), "Linux ggml-base static library");
            RequireThirdPartyFile(Path.Combine(LinuxLibPath, "libcpp-httplib.a"), "Linux cpp-httplib static library");
            PublicAdditionalLibraries.Add(LinuxArchive);
            PublicSystemLibraryPaths.Add(LinuxLibPath);
            PublicSystemLibraries.AddRange(new string[]
            {
                "astral_rt",
                "llama",
                "llama-common",
                "llama-common-base",
                "ggml",
                "ggml-cpu",
                "ggml-base",
                "cpp-httplib"
            });
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            PublicAdditionalLibraries.Add(RequireThirdPartyFile(
                Path.Combine(LibPath, "Mac", "libastral_rt.a"),
                "Mac static library"
            ));
        }
        else if (Target.Platform == UnrealTargetPlatform.Android)
        {
            PublicAdditionalLibraries.Add(RequireThirdPartyFile(
                Path.Combine(LibPath, "Android", "arm64-v8a", "libastral_rt.a"),
                "Android arm64 static library"
            ));
        }
        else if (Target.Platform == UnrealTargetPlatform.IOS)
        {
            PublicAdditionalLibraries.Add(RequireThirdPartyFile(
                Path.Combine(LibPath, "IOS", "libastral_rt.a"),
                "IOS static library"
            ));
        }
        else
        {
            throw new BuildException("AstralRT does not ship a native library for Unreal platform {0}", Target.Platform);
        }

        bEnableExceptions = false;
        bUseRTTI = false;
    }
}
