import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import configparser
import io
from azure_blob_service import upload_file_to_blob

# Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

# 設定 Azure Speech
speech_key = config.get('Azure', 'AZURE_SPEECH_KEY')
speech_region = config.get('Azure', 'AZURE_REGION')
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)

# Azure TextAnalytics
text_key = config.get('Azure', 'AZURE_TEXT_KEY')
text_endpoint = config.get('Azure', 'AZURE_TEXT_ENDPOINT')
text_client = TextAnalyticsClient(endpoint=text_endpoint, credential=AzureKeyCredential(text_key))

# Azure Text to Speech 
def text_to_speech(text, voice_type, user_id):
    speech_config.speech_synthesis_voice_name = voice_type 
    try:
        # 配置音訊輸出
        # audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_name)
        pull_stream = speechsdk.audio.PullAudioOutputStream()
        audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        # 進行語音合成
        result = synthesizer.speak_text_async(text).get()

        if not result:
            print("[ERROR] TTS result is None.")

        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            raise Exception(f"Speech synthesis canceled: {cancellation_details.reason}")
        try:
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                duration = int(result.audio_duration.total_seconds() * 1000)  # 轉為毫秒
                print(f"Audio duration: {duration} ms")
            else:
                raise Exception("Failed to get audio duration")
        except Exception as e:
            print(f"Error calculating audio duration: {e}")
            
        # 上傳檔案至 Azure Blob Storage
        audio_bytes_io = io.BytesIO()
        audio_bytes_io.write(result.audio_data)
        audio_bytes_io.seek(0)  # 重新定位到開始位置
        if not audio_bytes_io:
            raise Exception("Failed to retrieve audio data from PullAudioOutputStream.")
        blob_url = upload_file_to_blob(audio_bytes_io, user_id, "wav")
        if not blob_url:
            raise Exception("Blob upload failed.")
        print(f"Audio file uploaded to: {blob_url}")
        return blob_url, duration
    
    except Exception as e:
        raise Exception(f"Error during text-to-speech: {e}")
    
# Azure Text Analytics
def language_detection(text):
    try:
        response = text_client.detect_language(documents=text)[0]

        language_name = response.primary_language.name  # 語言名稱
        language_code = response.primary_language.iso6391_name  # 語言代碼

        return language_name, language_code

    except Exception as err:
        print(f"Encountered exception: {err}")
        return "unknown", "unknown"

if __name__ == "__main__":
    # text = "你好，這是一個測試"
    # user_id = "test"
    # audio_url, audio_duration = text_to_speech(text, user_id)
    # print(f"Audio URL: {audio_url}")
    # print(f"Audio duration: {audio_duration} ms")
    
    documents = ["Ce document est rédigé en Français."]
    aa = ['好']
    bb = ['Pork XiaoLongBao, Drus Roe and Pork XiaoLongBao, Chicken XiaoLongBao, Green Squash and Shrimp XiaoLongBao, Truffle and Pork XiaoLongBao']

    a, b = language_detection(bb)
    print(a, b)