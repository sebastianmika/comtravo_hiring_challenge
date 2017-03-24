import sys


def print_request(r):
    """Print a request in a human readable format to stdout"""
    def fmt_token(t):
        return t['shape'] + t['after']

    print('Subject: ' + ''.join(map(fmt_token, filter(
        lambda x: x['where'] == 'subject', r['tokens']))))
    print(''.join(map(fmt_token, filter(
        lambda x: x['where'] == 'body', r['tokens']))))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("predict_category <train_file> <test_file>")

    train_file, test_file = sys.argv[1:]
    # ToDo: Implement logic to find a model based on data in train_file

    # ToDo: Make predictions on data in test_file

    # ToDo: Generate output
