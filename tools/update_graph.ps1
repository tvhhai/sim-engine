# Tạo lại graph cục bộ, không gọi mô hình hay dịch vụ API.
param([switch]$Html)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    graphify extract . --code-only --no-cluster --max-workers 4
    if ($LASTEXITCODE -ne 0) { throw 'Không trích xuất được graph.' }
    graphify cluster-only . --no-label --no-viz
    if ($LASTEXITCODE -ne 0) { throw 'Không phân cụm được graph.' }
    if ($Html) {
        graphify export html
        if ($LASTEXITCODE -ne 0) { throw 'Không xuất được HTML.' }
    }
} finally {
    Pop-Location
}
