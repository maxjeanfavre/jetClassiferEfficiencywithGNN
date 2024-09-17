# Définir le répertoire contenant les fichiers Python
directory="/work/mjeanfav/jetClassiferEfficiencywithGNN"

# Ancien et nouveau nom de la variable
old_name="Jet_btagDeepB"
new_name="Jet_btagDeepFlavB"

# Use find to browse all Python files recursively
find "$directory" -type f -name "*.py" | while IFS= read -r fichier; do
    # Use sed to replace the variable name
    sed -i "s/\b$old_name\b/$new_name/g" "$fichier"
done

echo "Modification done."
