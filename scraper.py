import asyncio
from twikit import Client
import config
import pandas as pd
async def main():
    client = Client(
    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15'
)
    await client.login(
        auth_info_1=config.USERNAME,
        auth_info_2=config.EMAIL,
        password=config.PASSWORD
    )
    tweets = await client.search_tweet('Trump','Media',50)
    tweet_text_list = []
    for t in tweets:
        tweet_text_list.append(t.text)
    data_frame = pd.DataFrame(tweet_text_list)
    data_frame.to_csv('outputs.csv')
    
asyncio.run(main())