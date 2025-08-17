class Counter:
    def __init__(self):
        self.stats = {
            "Pass": 0,
            "LowRes": 0,
            "NoFace": 0,
            "MultiFace": 0,
            "NotIMG": 0,
            "Errors": 0
        }

    def increment(self, category):
        if category in self.stats:
            self.stats[category] += 1

    def get_stats(self):
        return self.stats

    def get_total_fails(self):
        return sum(v for k, v in self.stats.items() if k != "Pass")

    def get_summary(self):
        summary = "\n"
        summary += "="*30 + "\n"
        summary += "작업이 완료되었습니다.\n"
        summary += f"총 소요 시간: {self.total_time}\n"
        summary += "="*30 + "\n"
        summary += "분류 결과 통계:\n"
        summary += f"  - Pass: {self.stats['Pass']}개\n"
        summary += f"  - Fail (Low Resolution): {self.stats['LowRes']}개\n"
        summary += f"  - Fail (No Face Detected): {self.stats['NoFace']}개\n"
        summary += f"  - Fail (Multiple Faces Detected): {self.stats['MultiFace']}개\n"
        summary += f"  - Fail (Not an Image File): {self.stats['NotIMG']}개\n"
        if self.stats['Errors'] > 0:
            summary += f"  - Errors (Processing Failed): {self.stats['Errors']}개\n"
        summary += "="*30
        return summary
        
    def set_total_time(self, total_time):
        import time
        self.total_time = time.strftime('%H:%M:%S', time.gmtime(total_time))