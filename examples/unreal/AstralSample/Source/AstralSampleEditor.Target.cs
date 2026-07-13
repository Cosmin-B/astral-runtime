using UnrealBuildTool;
using System.Collections.Generic;

public class AstralSampleEditorTarget : TargetRules
{
    public AstralSampleEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
        ExtraModuleNames.Add("AstralSample");
    }
}
