from typing import List

class BaseModel():
    def __init__(self, url: str='',
                 title : str='',
                 content: List[str]=[],
                 *args, **kwargs):
        self.url = url
        self.title = title
        self.content = content

class DocText(BaseModel):
    def __init__(self, url: str = '',
                 title: str = '',
                 content: List[str] = [],
                 *args, **kwargs):
        super(DocText, self).__init__(url, title, content, *args, **kwargs)

class SectionText(BaseModel):
    def __init__(self, url: str = '',
                 title: str = '',
                 content: List[str] = [],
                 *args, **kwargs):
        super(SectionText, self).__init__(url, title, content, *args, **kwargs)

class ElasticResults():
    def __init__(self, doc: DocText, section: SectionText, *args, **kwargs):
        super(ElasticResults, self).__init__(*args, **kwargs)
        self.doc = doc
        self.section = section

    @property
    def doc_content(self):
        return self.doc.content

    @property
    def doc_url(self):
        return self.doc.url

    @property
    def section_content(self):
        return self.section.content

    @property
    def section_url(self):
        return self.section.url

def document_dict2obj(d):
    ## Todo: fill
    pass
    # sections = []
    # for i in d['content']:
    #     SectionText(url=, title=, content=)

def section_dict2obj(d):
    ## Todo: fill
    pass
    # sections = []
    # for i in d['content']:
    #     SectionText(url=, title=, content=)