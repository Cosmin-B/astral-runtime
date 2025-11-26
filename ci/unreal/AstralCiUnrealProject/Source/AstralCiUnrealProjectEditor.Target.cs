using UnrealBuildTool;
using System.Collections.Generic;

public class AstralCiUnrealProjectEditorTarget : TargetRules
{
    public AstralCiUnrealProjectEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        ExtraModuleNames.Add("AstralCiUnrealProject");
    }
}
