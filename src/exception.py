import sys
from logger import logger  


def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb.tb_frame is not None and exc_tb.tb_lineno is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occured in Python script '{0}' at line number {1}: {2}".format(
            file_name, exc_tb.tb_lineno, str(error)
        )
    else:
        error_message = "Error occurred: {0}".format(str(error))
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail)
    
    def __str__(self):
        return self.error_message


# Check if above code works
if __name__ == "__main__":
    try:
        x = 1/0
    except Exception as e:
        logger.info("Divide by Zero")
        raise CustomException(e, sys)