using UnrealBuildTool;

public class AstralSample : ModuleRules
{
    public AstralSample(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "AstralRT"
        });
    }
}
