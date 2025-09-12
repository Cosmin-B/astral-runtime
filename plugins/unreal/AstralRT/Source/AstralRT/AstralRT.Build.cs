using UnrealBuildTool;
using System.IO;

public class AstralRT : ModuleRules
{
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
        string ThirdPartyPath = Path.Combine(ModuleDirectory, "../../ThirdParty/AstralCore");
        string IncludePath = Path.Combine(ThirdPartyPath, "include");
        string LibPath = Path.Combine(ThirdPartyPath, "lib");

        PublicIncludePaths.Add(IncludePath);

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string StaticLib = Path.Combine(LibPath, "Win64", "astral_rt.lib");
            string StaticLibAlt = Path.Combine(LibPath, "Win64", "astral_rt_static.lib");
            PublicAdditionalLibraries.Add(File.Exists(StaticLib) ? StaticLib : StaticLibAlt);
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            PublicAdditionalLibraries.Add(Path.Combine(LibPath, "Linux", "libastral_rt.a"));
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            PublicAdditionalLibraries.Add(Path.Combine(LibPath, "Mac", "libastral_rt.a"));
        }
        else if (Target.Platform == UnrealTargetPlatform.Android)
        {
            PublicAdditionalLibraries.Add(Path.Combine(LibPath, "Android", "arm64-v8a", "libastral_rt.a"));
        }
        else if (Target.Platform == UnrealTargetPlatform.IOS)
        {
            PublicAdditionalLibraries.Add(Path.Combine(LibPath, "IOS", "libastral_rt.a"));
        }

        bEnableExceptions = false;
        bUseRTTI = false;
    }
}
