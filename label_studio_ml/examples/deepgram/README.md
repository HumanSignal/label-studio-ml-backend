
# Using Deepgram with Label Studio for Text to Speech annotation

This backend uses the Deepgram API to take the input text from the user, do text to speech, and return the output audio for annotation in Label Studio.

https://github.com/user-attachments/assets/9569a955-0baf-4a95-9e8a-d08250a0a298


IMPORTANT NOTE: YOU MUST REFRESH THE PAGE AFTER SUBMITTING THE TEXT TO SEE THE AUDIO APPEAR.

## Prerequistes 
1. [Deepgram API Key](https://deepgram.com/) -- create an account and follow the instructions to get an api key with default permissions. Store this key as `DEEPGRAM_API_KEY` in `docker_compose.yml`
2. AWS Storage -- make sure you configure the following parameters in `docker_compose.yml`: 
      - `AWS_ACCESS_KEY_ID` -- your AWS access key id
      - `AWS_SECRET_ACCESS_KEY` -- your AWS secret access key
      - `AWS_SESSION_TOKEN` -- your AWS session token
      - `AWS_DEFAULT_REGION` - the region you want to use for S3
      - `S3_BUCKET` -- the name of the bucket where you'd like to store the created audio files
      - `S3_FOLDER` -- the name of the folder within the specified bucket where you'd like to store the audio files. 
3. Label Studio -- make sure you set your `LABEL_STUDIO_URL` and your `LABEL_STUDIO_API_KEY` in `docker_compose.yml`. As of 11/12/25, you must use the LEGACY TOKEN. 

## Labeling Config 
This is the base labeling config to be used with this backend. Note that you may add additional annotations to the document after the audio without breaking anything!
```
<View>
  <Header value="What would you like to TTS?"/>
  <TextArea name="text" toName="audio" placeholder="What do you want to tts?" value="$text" valrows="4" maxSubmissions="1"/>
  <Audio name="audio" value="$audio" zoom="true" hotkey="ctrl+enter"/>
</View>
```
## A Data Note 
Note that in order for this to work, you need to upload dummy data (i.e. empty text and audio) so that the tasks populate. You can use `dummy_data.json` as this data. 

## Configuring the backend 
When you attach the model to Label Studio in your model settings, make sure to toggle ON interactive preannotations! 
