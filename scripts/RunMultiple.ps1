$interpreterPath = Read-Host "Enter interpreter path (python.exe)"
$scriptPath = Read-Host "Enter script path"
$numRuns = [int] (Read-Host "Enter number of runs")

foreach($i in 1..$numRuns){
	& $interpreterPath $scriptPath
}


Read-Host "Done. Press Enter to close.."