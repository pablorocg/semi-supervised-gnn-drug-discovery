class ToFloatTransform:
    """Converts node features `data.x` to float."""

    def __call__(self, data):
        data.x = data.x.float()
        return data