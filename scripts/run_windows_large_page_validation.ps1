param(
    [string]$Preset = "release-with-tests",
    [string]$OutDir = "build/windows-large-pages",
    [switch]$ExpectLargePages,
    [switch]$ExpectFallback
)

$ErrorActionPreference = "Stop"

if ($ExpectLargePages -and $ExpectFallback) {
    throw "Use only one of -ExpectLargePages or -ExpectFallback."
}

if (-not $IsWindows) {
    throw "Windows large-page validation requires a Windows host."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir "windows-large-pages.log"
$priv = Join-Path $OutDir "windows-privileges.txt"

function Invoke-Logged {
    param(
        [string]$Title,
        [string]$Command,
        [string[]]$Arguments
    )

    Add-Content -Path $log -Value ""
    Add-Content -Path $log -Value "[$Title] $Command $($Arguments -join ' ')"
    & $Command @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed with exit code $LASTEXITCODE"
    }
}

"Astral Windows large-page validation" | Set-Content -Path $log
"Timestamp: $((Get-Date).ToString('o'))" | Add-Content -Path $log
"Preset: $Preset" | Add-Content -Path $log
"ExpectLargePages: $ExpectLargePages" | Add-Content -Path $log
"ExpectFallback: $ExpectFallback" | Add-Content -Path $log
"OS: $([System.Environment]::OSVersion.VersionString)" | Add-Content -Path $log
"PowerShell: $($PSVersionTable.PSVersion)" | Add-Content -Path $log
"cmake: $((cmake --version | Select-Object -First 1) -replace '\r', '')" | Add-Content -Path $log
"ctest: $((ctest --version | Select-Object -First 1) -replace '\r', '')" | Add-Content -Path $log

whoami /all | Tee-Object -FilePath $priv | Out-Null
"Privilege report: $priv" | Add-Content -Path $log

if ($ExpectLargePages) {
    $env:ASTRAL_TEST_EXPECT_LARGE_PAGES = "1"
    [Environment]::SetEnvironmentVariable("ASTRAL_TEST_EXPECT_LARGE_PAGE_FALLBACK", $null, "Process")
} elseif ($ExpectFallback) {
    $env:ASTRAL_TEST_EXPECT_LARGE_PAGE_FALLBACK = "1"
    [Environment]::SetEnvironmentVariable("ASTRAL_TEST_EXPECT_LARGE_PAGES", $null, "Process")
} else {
    [Environment]::SetEnvironmentVariable("ASTRAL_TEST_EXPECT_LARGE_PAGES", $null, "Process")
    [Environment]::SetEnvironmentVariable("ASTRAL_TEST_EXPECT_LARGE_PAGE_FALLBACK", $null, "Process")
}

Invoke-Logged "configure" "cmake" @("--preset", $Preset)
Invoke-Logged "build" "cmake" @("--build", "--preset", $Preset, "-j")
Invoke-Logged "focused-ctest" "ctest" @("--preset", $Preset, "-R", "^(test_platform|test_core)$", "-V", "--output-on-failure")

"[windows-large-pages] OK: $log"
