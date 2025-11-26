using UnrealBuildTool;
using System.Collections.Generic;

public class AstralCiUnrealProjectTarget : TargetRules
{
    public AstralCiUnrealProjectTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        ExtraModuleNames.Add("AstralCiUnrealProject");
    }
}
