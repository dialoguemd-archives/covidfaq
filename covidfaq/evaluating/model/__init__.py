def build_isolator(chars):
    def isolate(text):
        for c in chars:
            text = text.replace(c, f" {c} ")
        return text
    return isolate