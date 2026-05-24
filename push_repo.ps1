git add .gitignore
git commit -m "updated .gitignore"

$files = git ls-files --others --exclude-standard
foreach ($file in $files) {
    git add $file
    $basename = Split-Path $file -Leaf
    git commit -m "updated $basename"
}

git branch -M main
git push -u origin main
