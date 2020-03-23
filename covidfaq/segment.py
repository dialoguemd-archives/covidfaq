import analytics
import structlog

log = structlog.get_logger()


def initialize(url, key):
    log.info("initializing segment", url=url)
    analytics.write_key = key
    analytics.send = key and key != "disable"
    analytics.host = url
    analytics.on_error = on_error


def track(event, data):
    analytics.track("12345566", event, data)


def on_error(error, items):
    log.warning("analytics failed", error=error)
