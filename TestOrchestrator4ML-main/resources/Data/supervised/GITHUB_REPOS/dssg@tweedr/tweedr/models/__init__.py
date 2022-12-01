from sqlalchemy import orm
from tweedr.lib.text import token_re
from tweedr.models.metadata import engine

sessionmaker = orm.sessionmaker(bind=engine)
DBSession = sessionmaker()

# we write enhanced ORM classes directly on top of the schema originals,
# so that enhancements are optional and transparent
from tweedr.models.schema import (
    DamageClassification,
    TokenizedLabel,
    UniformSample,
    Label,
    KeywordSample,
    Tweet,
)

# This quiets the 'import unused' pyflakes warning
__all__ = ['DamageClassification', 'TokenizedLabel', 'UniformSample', 'Label', 'KeywordSample', 'Tweet']


class DamageClassification(DamageClassification):
    # DamageClassification does not actually have a single FK, but references multiple tables.
    # so this join is actually way more complicated
    # tweet_object = orm.relationship(Tweet, lazy='join',
        # primaryjoin=orm.foreign(DamageClassification.DSSG_ID) == Tweet.dssg_id)

    @property
    def text(self):
        'Run join query to get text of this labeled tweet'
        # FIXME: Slow. Consider join in instead.
        if not hasattr(self, 'text_'):
            if 'uniform' in self.which_sample:
                self.text_ = DBSession.query(UniformSample.text).filter(UniformSample.dssg_id == self.DSSG_ID).first()[0]
            elif 'keyword' in self.which_sample:
                self.text_ = DBSession.query(KeywordSample.text).filter(KeywordSample.dssg_id == self.DSSG_ID).first()[0]
            else:
                self.text_ = None
        return self.text_

    @property
    def label(self):
        if self.Infrastructure == 1 or self.Casualty == 1:
            return 1.
        else:
            return 0.


class TokenizedLabel(TokenizedLabel):
    # the FK is called token_type, even though it is an identifying key and not
    # an actual token type, so we name the target object "token_type_object"
    token_type_object = orm.relationship(Label, lazy='subquery',
        primaryjoin=orm.foreign(TokenizedLabel.token_type) == Label.id)

    @property
    def tokens(self):
        return token_re.findall(unicode(self.tweet).encode('utf8'))

    @property
    def labels(self, null_label=None):
        labels = []
        label_start, label_end = self.token_start, self.token_end
        for match in token_re.finditer(self.tweet):
            token_start, token_end = match.span()
            # token = match.group(0)
            # we want to determine if this particular token in the original tweet overlaps
            #   with any portion of the selected label (label_span)
            label = null_label
            if label_start <= token_start <= label_end or label_start <= token_end <= label_end:
                label = self.token_type_object.text
            labels.append(label)
        return [unicode(label).encode('utf8') for label in labels]
