from atomworks.ml.datasets.parsers import GenericDFParser


class GenericDFParserCompat(GenericDFParser):
    """Compatibility wrapper for configs that inherit AF3 parser kwargs."""

    def __init__(
        self,
        *,
        base_dir: str | None = None,
        file_extension: str | None = None,
        path_template: str | None = None,
        **kwargs,
    ):
        if base_dir is not None and "base_path" not in kwargs:
            kwargs["base_path"] = base_dir
        if file_extension is not None and "extension" not in kwargs:
            kwargs["extension"] = file_extension
        super().__init__(**kwargs)
