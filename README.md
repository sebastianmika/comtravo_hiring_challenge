# Comtravo NLP/ML Hiring Challenge

As part of our machine learning team recruitment process we would be happy, if you complete the following challenge. It is not mandatory but will be much appreciated as it help us to get a better idea of your strengths.

## The Challenge

At Comtravo we work to improve the business travel planing and booking experience by providing our customers with quick turnaround, high quality offers for their requests. Our customer interface is text, e.g. e-mail.

A fundamental building block is to be able to correctly bucket the customer requests into the following categories:

1. Booking requests: someone asking to book something, e.g. a flight
2. Cancellation request: someone asking to cancel an existing booking
3. Rebooking request: someone asking to change an existing booking, e.g. fly on another day
4. Negotiation request: someone asking to change an offer, e.g. add a piece of luggage or leave on another day (before the booking was confirmed)
5. Other requests: someone wants something else, e.g. update its customer data or this might just as well be spam
6. ...and all combinations thereof

In this challenge you are given real e-mail data from our customer requests. The data is already tokenized and - for privacy reasons - you only receive meta information for each token. Your task is to map each request to its corresponding categories, where the distinction `booking` vs. anything else is most important. Obviously there is relevant information missing in this dataset, but we can assure you that it is still possible to find a relatively good solution!


### Dataset

The dataset in `comtravo_challenge_train.json` and `comtravo_challenge_test.json` consist of 7571 training samples and 1893 test samples. Each sample is a mapping with the following format:

```python
{
        'id': 'f39870dd04defade3f57aa21a5e185ea' # an unique request identifier
        'labels': {'booking': 1.0, 'other': 1.0} # a mapping of categories to probabilities for
                                                 # that category (all 1.0 for the training data,        
                                                 # categories not present score 0.0);
                                                 # missing for the test samples
        'tokens': [...]                          # list of tokens, see below
}
``` 

Each token is of the following format:

```python
{
        'where': 'body',     # Where in the request was the token: subject or body?
        'start': 3,          # Start index in the message part
        'length': 5,         # Length of the token
        'shape': 'Xxxxx',    # Shape of the token
        'after': ' ',        # Character after the token
        'rner': 'B-Location' # IOB-coded entity type of the token; see below
}
```

The field `rner` contains information about whether the underlying part of text is describing one of the following entities:

| Type | Meaning | Examples |
| ---- | ------- | ------- |
| `AirlineOrAlliance` | Mention of an airline or airline alliance | `Lufthansa`, `FR`, `Star Alliance` |
| `Breakfast` | Mention of whether to have a breakfast included with a hotel room or not | with `breakfast` |
| `CarOption` | Mention of a rental car option | with `automatic transmission` |
| `CarRentalName` | Mention of a car rental company | `Sixt` |
| `CategoryOrClass` | Mention of e.g. a booking category or car class | in `business class`, `VW Passat`, `4 stars or above` |
| `Count` | Mention of a general count | `two` |
| `CountPersons` | Mention of a person/traveler count | book `two tickets` |
| `CountRooms` | Mention of a room count | book `three rooms` |
| `EndOfMsg` | Anything including and behind the message sign-off | starting at e.g. `Best Regards, ...` |
| `FlightNumber` | Mention of a flight number | `LH432` |
| `Greeting` | Anything in a greeting phrase | `Hi Comtravo,`, `Good morning,` |
| `HotelName` | Mention of a hotel name | `Adlon Berlin` |
| `Location` | Mention of a location | `Berlin`, `TXL`, `London-Heathrow` |
| `LoyaltyCard` | Mention of a loyalty card | `skymiles 1234 5678 9101` |
| `Luggage` | Mention of whether to include luggage or not | without `luggage` |
| `Meal` | Mention of specific meal requirements | `vegan meal` |
| `OtherEntity` | Mention of any other booking relevant entity | |
| `OtherOption` | Mention of any other booking relevant option | `free wifi` |
| `Person` | Mention of a person | `Mr. Schmidt`, `Max Müller` |
| `PriceLimit` | Mention of a price (limit) | less than `50EUR` |
| `Seat` | Mention of a seat preference | `seat 14E`, `window seat` |
| `TimePoint` | Mention of a time point | `Feb. 6th`, `next Monday`, `10:45` |   
| `TimeRange` | Mention of a time range | `10:30 - 12:15`, `before noon`, `next week` |

Each type is preceded by a `B-` for the start token of that entity and an `I-` while the entity continues on the next token. A value of just `O-` indicates that the token does not belong to any of the above groups.

#### Example Request:

To give you a slightly more specific idea what the original requests were like, here is an example of a (very short) `body` part:

| Text | Please | book | a | ticket | to | London | next | Monday | . |
| :--: | :----: | :--: | :--: | :----: | :----: | :--------: | :-----: | :--------: | :---: |
| shape | `Xxxxx` | `xxxx` | `x` | `xxxx` | `xx` | `Xxxxx` | `xxxx` | `Xxxxx` | `.` |
| start | 0 | 7 | 12 | 14 | 21 | 24 | 31 | 36 | 42 |
| length | 6 | 4 | 1 | 6 | 2 | 6 | 4 | 6 | 1 |
| after | `' '` | `' '` | `' '` | `' '` | `' '` | `' '` | `' '` | `''` | `''` |
| ner | `O-` | `O-` | `O-` | `O-` | `O-` | `B-Location` | `O-` | `B-TimePoint` | `O-` |

The correct prediction could be e.g. `{'booking': 0.98}`.

The script skeleton contains a small helper `print_request()` which demonstrates how to reconstruct the original request structure from the token sequence. An example for the first training sample yields this result:

```
Subject: Xxxxx Xxxx XX ddd xx dd.dd.dddd xxx dd:ddxx
Xxxxx Xxxxx-Xxxx,

xxxx xxx xxxx xxx Xxxxx Xxxxx xxx Xxxxx xxxx:

Xxxx Xxxxx > Xxxxx Xxxxx dd.d.dd
Xxxx Xxxxx > Xxxxx Xxxxx dd.d.dd xxxx

Xxxxx xx Xxxxx xx xxxx x) xxxx xxxx xxxx Xxxxx xxxx xxxx x) Xxxxx

Xxxxx Xxxx!


 <xxxx://xxxx.xx/>
Xxxxx Xxxxx

Xxxxx Xxxxx | xxx.xxxx.xx <xxxx://ddddXXX-XXdd-ddXX-dddX-XXXdddXddXdd/xxxx.xx>
xxxx@xxxx.xx <xxxx:xxxx@xxxx.xx> | +dd ddd dddd
 <xxxx://xxxx.xxx/xxxx/xxxx>  <xxxx://xxxx.xxx/xxxx>  <xxxx://xxxx.xxx/XXXXxXxxxx>

XXXXx XxxX | Xxxxx (Xxxx) | Xxxxx: Xxxxx ddd | dddd Xxxxx
Xxxxx xxxx Xxxxx Xxxxx xxx Xx. Xxxxx Xxxxx | Xxxxx Xxxxx XXX dddd X
xxxx.xxx
```

### Task
Your task is to fill the missing `labels` fields in the test data, i.e. make predictions to which category or categories a request in the test set belongs. The label format should be as in the training data and hence might also reflect the certainty of the category assignment.

More specifically, you should write a script `predict_category.py comtravo_challenge_train.json comtravo_challenge_test.json`, i.e. it can be called with the path to the provided training and test data files. The script should generate an output file named `comtravo_predictions.json` in the same format as the training data. A very rudimentary stub for `predict_category.py` is in the repo.

You are free to choose whatever approach you like. However, your solution should execute in **Python 3.5**. If you use any external libraries, please add a `requirements.txt` file that - when run with `pip install -r requirements.txt` in a fresh virtual environment - provides all required dependencies. You might want to use a virtual environment and when you are done call `pip freeze > requirements.txt`.

Please send us **all code required** to solve the task, enabling us to re-create your `comtravo_predictions.json`.

In addition to the code which solves the task please provide a short text file or pdf with **answers to the following questions**:

1. Your approach: how did you attack the problem, which methods did you chose and why?
2. How do you evaluate your performance and how well do you perform?
3. How could your approach be improved when e.g. spending more time or having access to the full request data?

Feedback to the challenge itself is very much appreciated as well.

### General Guidelines

Please send your solution and comments (or a link where to find them) as a reply to your challenge invitation. There you can also ask questions should they arise.

Please also read the challenge invitation carefully, it might contains additional or updated information and an individual deadline.

We are aware the challenges can be time consuming and we do not expect a perfect solution. Our primary interest is to learn how you tackle the problem and how you code. It is fine if you do not implement everything you could, but rather deliver a basic solution and outline what an improved solution would include in the answers to above questions.

We hope you will have as much fun in solving the challenge as we had in creating it!

Best,
The Comtravo NLP & ML Team!
