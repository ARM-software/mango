from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import re
import json


def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


def get_datalink(task_id):
    raw_html = simple_get(f'https://www.openml.org/t/{task_id}')
    html = BeautifulSoup(raw_html, 'html.parser')
    divs = html.findAll('div', attrs={'class': 'datainfo'})
    for div in divs:
        for a in div.findAll('a'):
            link = a['href']
            if re.match("^d\/", link):
                return link


def download_links(datalink):
    raw_html = simple_get(f'https://www.openml.org/{datalink}')
    html = BeautifulSoup(raw_html, 'html.parser')
    elems = html.findAll('a', attrs={'class': 'btn btn-link'})
    csv_url = json_url = None
    for elem in elems:
        link = elem['href']
        if re.match(".*\/get_csv\/.*", link):
            csv_url = link
        elif re.match(".*\/json$", link):
            json_url = link

    return csv_url, json_url


if __name__ == "__main__":
    # ref: https://arxiv.org/pdf/1909.12552.pdf
    # better ref using same experiments: https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning

    svm_taskids = [10101, 145878, 146064, 14951, 34536, 34537, 3485,
                   3492, 3493, 3494, 37, 3889, 3891, 3899, 3902, 3903,
                   3913, 3918, 3950, 6566, 9889, 9914, 9946, 9952, 9967,
                   9971, 9976, 9978, 9980, 9983]

    xgb_taskids = [10093, 10101, 125923, 145847, 145857, 145862, 145872,
                   145878, 145953, 145972, 145976, 145979, 146064, 14951,
                   31, 3485, 3492, 3493, 37, 3896, 3903, 3913, 3917, 3918,
                   3, 49, 9914, 9946, 9952, 9967]

    rf_taskids = [125923, 145804, 145836, 145839, 145855, 145862, 145878,
                  145972, 145976, 146065, 31, 3492, 3493, 37, 3896, 3902,
                  3913, 3917, 3918, 3950, 3, 49, 9914, 9952, 9957, 9967,
                  9970, 9971, 9978, 9983]

    taskids = list(set(svm_taskids + xgb_taskids + rf_taskids))

    for taskid in taskids:

        datalink = get_datalink(taskid)
        csv_url, json_url = download_links(datalink)
        print(taskid)
        print(datalink)
        print(csv_url)
        print(json_url)

        # r = get(csv_url)
        # with open(f'openml/{taskid}.csv', 'wb') as f:
        #     f.write(r.content)

        resp = get(f'https://www.openml.org{json_url}')
        with open(f'openml/{taskid}.json', 'wb') as f:
            f.write(resp.content)
