netsh advfirewall firewall add rule name="Jupyter 8000" dir=in action=allow protocol=TCP localport=8000
docker stop -t 0 ai
docker rm ai
docker run --name ai -d  -p 8000:9999  -v C:\training\training_notebook:/opt/notebook caapi/ai
$url = 'http://localhost:8000'
try {
    Start-Process "chrome.exe" $url
}
catch {
    Start-Process $url
}