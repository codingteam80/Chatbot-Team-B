from utils.file_utils import get_all_documents
from utils.manifest import ManifestManager

documents = get_all_documents()

manifest = ManifestManager.build(documents)

ManifestManager.save(manifest)

print("Documents:", len(documents))
print("Manifest Entries:", len(manifest))