$chunkPrefix = "46_"
$chunkFiles = Get-ChildItem -Filter $chunkPrefix* -Path .

# 创建输出文件流
$sourcePath = "E:\youfile\Graph_46\merged_file.pt"
$fileStream = [System.IO.FileStream]::new($sourcePath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
try {
    # 按顺序合并文件
    foreach ($file in $chunkFiles) {
        $chunkData = [IO.File]::ReadAllBytes($file.FullName)
        $fileStream.Write($chunkData, 0, $chunkData.Length)
    }
} finally {
    $fileStream.Dispose()
}