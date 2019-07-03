from imgurpython import ImgurClient

from src.main.python.core.pathUtils import PathUtils

ID = str

class ImageUploader:
    def upload(self, name: str, imageFilePath: str) -> ID:
        pass


class ImageUploaderImpl(ImageUploader):
    def __init__(self):
        clientId = "ddcb46c893bf0aa"
        clientSecret = "77da4f8ac4598429e288f3d480cff579d08837a8"
        self.client = ImgurClient(clientId, clientSecret)
    def upload(self, name: str, imageFilePath: str) -> ID:
        config = {
            "album": None,
            "name": name,
            "title": name,
            "description": ""
        }
        image = self.client.upload_from_path(imageFilePath, config = config, anon = False)
        return image["id"]

if __name__ == "__main__":
    imageUploader = ImageUploaderImpl()
    image = imageUploader.upload(
        "tho - 975.jpg",
        PathUtils.join(PathUtils.getResourcePath(), "reconsiderImages", "Tho", "975.jpg")
    )
    print(image)