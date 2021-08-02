### This script extracts the lafan1 dataset gotten from the submodule and splits it into
###  Training- and Validation datasets at the paths expected from the code.
### 
### If the code is ever updated to expect new paths, please change the $TrainingDest and ValidationDest variables.

$TrainingDest = "$($PSScriptRoot)\lafan1\train"
$ValidationDest = "$($PSScriptRoot)\lafan1\val"
$LafanPath = "$($PSScriptRoot)\lafan1_module\lafan1\lafan1.zip"

# This is a list of every file belonging to the validation set (chosen arbitrarily)
$ValidationFileList = "aiming2_subject3.bvh","dance1_subject3.bvh","dance2_subject3.bvh","fallAndGetUp1_subject5.bvh","fight1_subject5.bvh","fightAndSports1_subject4.bvh","ground1_subject1.bvh","ground2_subject3.bvh","jumps1_subject5.bvh","multipleActions1_subject1.bvh","multipleActions1_subject3.bvh","obstacles1_subject1.bvh","obstacles2_subject2.bvh","obstacles3_subject4.bvh","obstacles6_subject1.bvh","push1_subject2.bvh","pushAndFall1_subject4.bvh","pushAndStumble1_subject3.bvh","run1_subject5.bvh","sprint1_subject4.bvh","walk2_subject1.bvh","walk3_subject3.bvh","walk3_subject5.bvh","walk4_subject1.bvh"

Write-Host "Extracting from: $($LafanPath) into $($TrainingDest)"
Expand-Archive $LafanPath $TrainingDest

Write-Host "Splitting dataset"

# Assignment to null to hide output
$null = New-Item $ValidationDest -ItemType Directory
Foreach ($file in $ValidationFileList)
{
	$filePath = "$($TrainingDest)\$($file)"
	Move-Item $filePath $ValidationDest
}

Write-Host -NoNewLine 'Done. Press any key to continue...';
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown');