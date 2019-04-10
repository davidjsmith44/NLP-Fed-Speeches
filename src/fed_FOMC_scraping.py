'''
    SCRAPING FEDERAL RESERVE SPEECHES

This file contains the functions to scrape the federal reserve web site for
speech details and text from 2006 through today. The speeches are stored in
a dataframe with the following columns
    date
    speaker
    title
    link    (to site for speech text)
    text    (speech stored as a string)

The Federal Reserve website contains a searchable of their speeches at:
    https://www.federalreserve.gov/newsevents/speeches.htm

The speeches themselves are separated by year and stored on their own web pages with links
to the path to the text file. For example:
    "/newsevents/speech/2019-speeches.htm"
    "/newsevents/speech/2018-speeches.htm"
    "/newsevents/speech/2006speech.htm"

'''

def find_all_press_releases(host, this_url, print_test=False):
    '''
    Takes the host and a url to the FOMC web site for monetary policy press releases
    and return links to the web site containing the text of the press releases.
    This function is used to create the list of all web sites that contain the individual speeches that
    need to be scraped.

    INPUTS:
        host        the host (for the Federal Reserve 'www.federalreserve.gov)
        this_url         the path to the speeches for a given year
        print_test  an optional field that will print out summary statistics

    OUTPUT:
        final_links    list of htm links to the actual speeches

    '''
    conn = HTTPSConnection(host = host)
    conn.request(method='GET', url = this_url)
    resp = conn.getresponse()
    body = resp.read()
    # check that we received the correct response code
    if resp.status != 200:
        print('Error from Web Site! Response code: ', resp.status)
    else:
        soup=BeautifulSoup(body, 'html.parser')
        event_list = soup.find('div', id='article')
        # creating the list of dates, titles, speakers and html articles from web page
        link_list = []
        for link in event_list.findAll('a', href=True):
            link_list.append(link.get('href'))

        # now we need to clean the link_list to remove pdf versions of the statements
        # The statements are often listed as two links, one to a web site and one to a pdf.
        keep_these = []
        for i in range(len(link_list)):
            this_href = link_list[i]
            if 'newsevents/pressreleases/monetary' in this_href:
                keep_these.append(i)

        htm_links = []
        for item in keep_these:
            htm_links.append(link_list[item])

        # the Federeal Resereve web site contains multiple html links on each date. I am interested in the
        # policy statement only, which is kept as a date reference then 'a.ht,'
        # Filtering non-statements out
        keep_these = []
        for i in range(len(htm_links)):
            this_href = htm_links[i]
            if 'a.htm' in this_href:
                keep_these.append(i)

        final_links = []
        for item in keep_these:
            final_links.append(htm_links[item])

        return final_links

def create_speech_df(date_list, doc_list):
    '''
    Creates dataframe containing from the date_list and doc_list
    of the Federal Open Market Committee press releases.
    This gets called after the 'find_all_press_releases' and 'retrieve_docs'
    functions have been called
    INPUTS:
        date_list       The dates of the FOMC speeches
        doc_list        The list of press release content

    OUTPUT:
        df    a dateframe containing the following columns
            ['date]         date of press release
            ['text']        press release content


    '''
    date_df = pd.to_datetime(date_list)
    df_dict = {'date':date_df, 'text':doc_list}
    df = pd.DataFrame(df_dict)

    return df

def retrieve_docs(host, link_list):
    '''
    This function takes a dataframe with the columns 'link' and 'text' and the host to
    the paths contained in the link column. The original dataframe is returned with
    the text of the scrapped speeches in the 'text' column as a string

    INPUTS:
        host          the host to the Federal Reserve
        link_list     a list 'link' which contains all of the speech paths to be scrapped.

    OUTPUTS:
        doc_list      the original dataframe is returned with the column 'text' populated
                with the text from the speeches
        doc_date
    '''
    doc_list = []
    date_list = []
    for i in range(len(link_list)):
        this_item = link_list[i]
        print('Scraping text for documents #: ', i)
        this_date, this_doc = get_one_doc(host, this_item)
        doc_list.append(this_doc)
        date_list.append(this_date)
    return doc_list, date_list

def get_one_doc(host, this_url):
    '''
    This function takes a host and url containing the location of one Federal
    Reserve speech and returns a string containing the text from the speech.

    INPUTS:
        host    the host to the Federal Reserve
        url     the path to a particular speech

    OUTPUTS:
        string  containing the text of the speech

    '''
    #conn = HTTPSConnection(host = host)
    #conn.request(method='GET', url = this_url)
    #response = conn.getresponse()

    temp_url = 'https://' + host + this_url
    response = requests.get(temp_url)
    sp = BeautifulSoup(response.text)
    this_date = sp.find('p', class_='article__time')
    this_date = this_date.text

    article = sp.find('div', class_='col-xs-12 col-sm-8 col-md-8')
    doc = []
    for p in article.find_all('p'):
        doc.append(p.text)

    return_doc = ''.join(doc)

    return this_date, return_doc

if __name__ == '__main__':

    # import functions
    import pandas as pd
    import numpy as np
    from bs4 import BeautifulSoup
    from http.client import HTTPSConnection
    import pickle
    from urllib.request import urlopen
    import requests
    import os

    host = 'www.federalreserve.gov'
    prefix = '/monetarypolicy/fomccalendars.htm'

    link_list =find_all_press_releases(host, prefix, print_test=False)

    doc_list, date_list = retrieve_docs(host, link_list)

    df = create_speech_df(date_list, doc_list)

    # # create dataframe containing speech information (not yet the text)
    # df = create_speech_df(host, annual_htm_list)
    # #print(df.info())
    # df['date'] = pd.to_datetime(df['date'])
    # df['date'] = df['date'].dt.strftime('%m/%d/%Y')

    # # scrape the text from every speech in the dataframe
    # df = retrieve_docs(host, df)
    # print(df.info())

    # saving the df to a pickle file
    #os.chdir("..")
    pickle_out = open('mvp_fed_press_rel', 'wb')
    pickle.dump(df, pickle_out)
    pickle_out.close()

