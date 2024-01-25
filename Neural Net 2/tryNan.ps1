$Snetworkshare = "E:\test"  # Enter the UNC path between the quotes
$date = Get-Date
$Domain = "bcc.qld.gov.au"  # Enter the domain name between the quotes
$subject = "DOWNLOADED FILES ON nk folder WILL BE DELETED SOON" # Enter the subject for the mail address

Function SendEmail([string] $ToEmail, [string] $subject, [string] $SMessage) {
    $smtpServer = "smtp.bcc.qld.gov.au"  # Enter the SMTP server
    $smtpFrom = "M638DEV1-SEGPO1 brisbane.qld.gov.au"  # Anonymous relay needs to be enabled on a Receive Connector
    $smtp = New-Object Net.Mail.SmtpClient($smtpserver)
    $smtp.Send($smtpFrom, $ToEmail, $subject, $SMessage)
}

$allfolders = Get-ChildItem -Recurse $Snetworkshare

foreach ($folder in $allfolders) {
    $foldername = $folder.Name
    $allitems = Get-ChildItem -Recurse $folder.FullName
    $MailItems = ""

    foreach ($item in $allitems) {
        if (-not $item.PSIsContainer) { # Check if the item is a file
            if ($date -lt $item.LastAccessTime.AddDays(-10)) { # Check if the file access time is older than 10 days
                $MailItems += "$($item.FullName)`n" # Append the file name to an email message
            }
        }

        if ($date -lt $item.LastAccessTime.AddDays(-15)) { # If the file access time is 15 days old or older
            if (-not $item.PSIsContainer) { # If the item is a file, remove it
                Remove-Item $item.FullName -Force
            }
        }
    }

    if ($MailItems -ne "") { # If there are files to notify about
        $message = "Dear $foldername,`n`nThe following items will be removed in 4 days:`n`n" + $MailItems
        $ToEmail = Get-ADUser -Filter { SamAccountName -eq $foldername } -Properties mail | Select-Object -ExpandProperty mail
        SendEmail -ToEmail $ToEmail -subject $subject -SMessage $message
    }

    do{
        # Get all the empty folders and subfolders, sorted by depth in descending order
        $sdirs = Get-ChildItem -Path $Snetworkshare -Directory -Recurse -Force | Where-Object { (Get-ChildItem -Path $_.FullName -Force).Count -eq 0 } | Sort-Object -Property FullName -Descending

        # Loop through each folder and delete it
        $sdirs | ForEach-Object {
            # Print the folder name
            Write-Host "Deleting folder '$($_.FullName)'"
            # Delete the folder, with confirmation and testing
            Remove-Item -Path $_.FullName -Confirm -WhatIf
        }
    }while($sdirs.Count -gt 0)
    
}
