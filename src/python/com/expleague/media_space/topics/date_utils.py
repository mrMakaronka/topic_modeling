from datetime import datetime


class DateUtils:
    @staticmethod
    def normalize(dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0)
