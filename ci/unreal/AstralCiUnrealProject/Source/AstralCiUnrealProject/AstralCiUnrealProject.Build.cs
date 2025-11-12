using UnrealBuildTool;

public class AstralCiUnrealProject : ModuleRules
{
    public AstralCiUnrealProject(ReadOnlyTargetRules Target) : base(Target)
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
