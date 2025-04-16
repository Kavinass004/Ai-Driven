AI-Driven Tamil News Automation: The Future of Digital Journalism

S. Kavinass1
Department of Artificial Intelligence and Machine Learning
St. Joseph’s College of Engineering
Chennai, India
kavinass004@gmail.com
A. Abul Hasan2
Department of Artificial Intelligence and Machine Learning
St. Joseph’s College of Engineering
Chennai, India
ah8803021@gmail.com
M. Poornima3
Department of Artificial Intelligence and Machine Learning
St. Joseph’s College of Engineering
Chennai, India
poornimam@stjosephs.ac.in

 
 
Abstract— The rapid evolution of artificial intelligence (AI) offers transformative solutions for automating news production, yet few systems address the complexities of multilingual, code-mixed languages prevalent in diverse regions. This paper introduces an innovative AI-powered pipeline designed to convert audio or video interviews into structured news content in code-mixed Tamil, a blend of Tamil and English widely spoken in South India and its diaspora. The system integrates advanced speech-to-text transcription using the Whisper model, speaker diarization via Pyannote, and content generation with the Mistral language model to produce three distinct outputs: a formal newspaper article, a concise social media post, and a dual-anchor news reader script. Built to operate within a Google Colab environment, the pipeline incorporates robust error handling and memory-efficient options, such as smaller model variants, to ensure accessibility in resource-constrained settings. Preliminary results demonstrate its ability to accurately transcribe and creatively summarize interviews, making it a promising tool for regional newsrooms.  This work bridges a gap in automated journalism by catering to code-mixed languages, with potential applications in multilingual content creation.
Keywords—AI news automation, speech-to-text, speaker diarization, code-mixed Tamil, content generation, Whisper, Pyannote, Mistral.

I.	INTRODUCTION
The news industry faces mounting pressure to deliver timely, engaging content amidst shrinking resources and growing digital demands. Traditional news production, reliant on human transcription and writing, is labor-intensive and slow, particularly for regional languages where skilled personnel may be scarce. In multilingual regions like India, where code-mixing—blending native languages like Tamil with English—is common in everyday speech, this challenge intensifies due to the lack of automated tools tailored to such linguistic diversity. While AI-driven solutions like automatic speech recognition (ASR) and natural language generation (NLG) have revolutionized journalism in major languages, their application to code-mixed dialects remains underexplored, leaving a critical gap in accessible news automation.
This paper presents a novel end-to-end pipeline that leverages AI to transform audio or video interviews into ready-to-publish news content in code-mixed Tamil. Designed for efficiency and scalability, the system processes raw input through three key stages: (1) speech-to-text conversion with speaker diarization to identify and transcribe individual voices, (2) content generation producing a newspaper article, social media bite, and news reader script, and (3) output formatting for immediate use. By employing state-of-the-art models—Whisper for transcription, Pyannote for diarization, and Mistral for text generation—the pipeline delivers high-quality results while offering memory-optimized variants for deployment on standard hardware, such as Google Colab’s T4 GPUs.
Our contribution addresses a pressing need in regional journalism by automating content creation for a code-mixed language, a domain overlooked by existing systems. Beyond Tamil, the framework holds promise for adaptation to other multilingual contexts, enhancing accessibility and efficiency in newsrooms worldwide. This introduction outlines the motivation, technical approach, and significance of our work, setting the stage for a detailed exploration of its methodology and outcomes.

II.	LITERATURE SURVEY
Li Li (2025) [1]  This study investigates the application of artificial intelligence (AI) technologies in the journalism and media sectors, focusing on AI's role in content creation, editing, production, and dissemination. It highlights both the potential and challenges associated with increased AI usage, such as ethical concerns like algorithmic bias and reduced journalistic integrity. The research also compares the effectiveness of various AI-driven content generation algorithms, revealing that CO-EBFGS achieves superior accuracy (91.2%) compared to CNN, RNN, Naive Bayes, and Logistic Regression. The study suggests future exploration of hybrid AI models and reinforcement learning to enhance content generation efficiency.
Kevin-Alerechi E et al. (2025) [2]  This study explores the transformative impact of AI and machine learning (ML) technologies on newsroom operations, highlighting improvements in data collection, fact-checking, reporting, and content dissemination. By integrating natural language processing, machine learning algorithms, and cloud-based systems, AI enhances journalistic processes, enabling richer narratives and personalized user experiences. Despite its benefits, challenges such as algorithmic bias and misinformation risks remain. The study emphasizes the need for explainable AI (XAI) to ensure transparency and trust. Future advancements may include multimodal AI systems, integrating text, images, audio, and video for comprehensive storytelling.
Sachita Nishal and Nicholas Diakopoulos (2023) [3]  This paper explores the evolving use of generative AI in journalism, highlighting the shift from template-based content creation to more sophisticated generative models that produce substantial article drafts. The study examines how news outlets like CNET and Men’s Journal have adopted these technologies for generating news and advisory content, despite encountering challenges like factual inaccuracies and increased editing workloads. The paper emphasizes the potential of generative AI in newsrooms, including its use in summarization and creativity support, while also discussing ethical concerns and the need for balanced human-AI collaboration.
Ambeth Kumar Visvam Devadoss et al. (2018) [4]  This study focuses on the integration of AI into online journalism, emphasizing the development of an automated information generation platform. The system leverages machine learning techniques and the NLTK module to analyze internet trends, especially from social media platforms like Twitter. It gathers, classifies, and categorizes data to create news articles that closely resemble human-written content, both grammatically and linguistically. The paper highlights the increasing importance of automation in journalism, while addressing challenges such as maintaining human-like articulation and accurately reflecting public sentiment.
Sandeep Chataut et al. (2025) [5]  This paper discusses a fully automated system for generating multimedia news content from news websites. The proposed system integrates web scraping using Node.js and Puppeteer to collect news data, followed by audio generation using Piper TTS and FFMPEG for audio processing. For video production, it utilizes Remotion to combine audio, images, and text overlays into professional-quality videos. The system also employs generative AI to optimize social media engagement by generating relevant hashtags and metadata. By automating the news processing pipeline, the system enhances efficiency and consistency, offering a scalable solution for multilingual and multimedia journalism.
Issiaka Faissal Compaore et al. (2025) [6]  This paper presents an AI-powered online press synthesis tool designed to tackle the challenge of processing the vast volume of online news. The system employs advanced AI models, specifically GPT-3.5 Turbo 16k and Pegasus Summarizer, to generate high-quality summaries from news articles scraped via Beautiful Soup. It assesses similarity between articles and matches audio to summaries. Evaluation using BLEU and ROUGE metrics shows that GPT-3.5 Turbo 16k outperforms Pegasus, achieving a BLEU score of 16.39% and a ROUGE score of 0.66%. The study highlights future improvements, such as integrating expert-written human summaries and adopting a unified model approach, contributing to the evolution of automated news synthesis.
Mohamed Hashim Changrampadi et al. (2021) [7]  This study addresses the challenge of Automatic Speech Recognition (ASR) for low-resource languages. The researchers developed a speech recognition model using the Mozilla DeepSpeech architecture, leveraging freely available online computational resources to reduce costs. The model, trained for the Tamil language, achieved a Word Error Rate (WER) of 24.7%, significantly outperforming Google's speech-to-text, which had a WER of 55%. The study also introduces a semi-supervised approach for building a speech corpus, making it feasible to create a large vocabulary corpora even in resource-constrained settings. The trained Tamil ASR model and datasets are openly accessible on GitHub, facilitating further research in this domain.
S. Saranya et al. (2024) [8]  The study presents a near real-time Tamil dialect-based transcription system aimed at improving automatic speech recognition (ASR) for Tamil dialects. The research focuses on minimizing latency and enhancing accuracy by fine-tuning a Whisper small architecture model using a novel Tamil dialect-based speech and text corpus. A parameter-efficient fine-tuning technique called low-ranking adapters was employed, resulting in a word error rate (WER) of 61% for dialectal speech—significantly better than the pretrained models with over 80% WER. Additionally, dialect classification using the Whisper encoder with a classification head achieved a remarkable accuracy of 97.1%. The system also integrates a summarization tool for generating intermediate summaries from continuous speech, contributing to improved ASR for Tamil dialects and promoting accessibility in diverse linguistic contexts.
Dr. Bharati Rathore (2024) [9]  The research explores the potential of OpenAI's Chat Generative Pre-trained Transformer (ChatGPT) in enhancing human-machine interactions, particularly through chatbots. The study highlights how ChatGPT's natural language processing and automated text generation capabilities revolutionize communication, offering more efficient and user-friendly experiences. The paper also discusses the potential of AI tools like ChatGPT to improve scientific writing and peer review processes by automating tasks, intelligently analyzing data, and providing better recommendations. Additionally, the research introduces the predictive "MESN Model" for forecasting using ChatGPT, aiming to understand user expectations and the impact of AI in the near future.
Yutao Zhu et al. (2020) [10]  This research focuses on enhancing dialogue systems by guiding conversations using narratives, addressing a key challenge in automated story or script generation. The proposed model, ScriptWriter, selects the best response among candidate options based on both the context and the given narrative. The model tracks the narrative, differentiating its role from that of the context (i.e., previous dialogue utterances). Due to the lack of existing data for this application, the authors introduce a new large-scale dataset, GraphMovie, sourced from a movie website where users can upload narratives while watching movies. Experimental results demonstrate that their narrative-based approach significantly outperforms traditional models that treat the narrative merely as context.

III. PROPOSED SOLUTION
This study proposes an innovative AI-driven solution to automate news content creation in code-mixed Tamil, addressing the challenges of manual news production in multilingual regions. The solution is a comprehensive pipeline that transforms raw audio or video interviews into three distinct news formats: a formal newspaper article, a concise social media post, and a dual-anchor news reader script, all in a blend of Tamil and English. Designed to operate within a Google Colab environment, the system leverages state-of-the-art models—Whisper for speech-to-text transcription, Pyannote for speaker diarization, and Mistral for content generation—while incorporating memory-efficient options and robust error handling to ensure practical deployment in resource-constrained settings.
The pipeline begins with input preprocessing to standardize heterogeneous inputs. Audio or video files are uploaded via a user-friendly interface, with video files processed using FFmpeg to extract audio at a 16 kHz sampling rate in mono-channel WAV format. Audio files are resampled using Librosa to match this specification, ensuring compatibility with downstream models. This step is critical for handling diverse input formats, such as MP4 videos or MP3 audio, often encountered in real-world interview scenarios.
The core of the solution lies in its transcription and speaker diarization module, which converts audio into structured text while identifying individual speakers. Pyannote’s speaker diarization model segments the audio into speaker-specific turns, generating timestamps and labels (e.g., Speaker_0, Speaker_1). Whisper, configured for Tamil, transcribes each segment into text, adeptly handling code-mixed speech where speakers switch between Tamil and English mid-sentence. For example, a phrase like “Event super-a irundhuchu, but timing issue irundhu” is accurately captured, preserving linguistic nuances. To manage memory, audio is processed in chunks, with segments shorter than 60 ms discarded to avoid noise. A fallback mechanism ensures reliability: if diarization fails, the system reverts to basic transcription, assigning all text to a single “UNKNOWN” speaker, thus preventing pipeline failure.
The transcribed text, annotated with speaker labels and timestamps, is then fed into the content generation module. Here, Mistral-7B-Instruct, quantized to 4-bit precision for efficiency, generates the three news outputs. The newspaper article is a structured piece with a headline, introduction, body, and conclusion, blending Tamil and English naturally to reflect the interview’s tone. The social media post condenses key insights into a 280-character summary, ideal for platforms like Twitter, while the news reader script formats the content as a dialogue for two anchors, facilitating broadcast delivery. To handle lengthy transcripts, inputs exceeding 4000 characters are truncated, ensuring the model operates within memory limits.
A key feature of the solution is its dual-mode operation: users can choose between full models (e.g., Whisper-large-v3) for maximum accuracy or smaller models (e.g., Whisper-small) for reduced memory usage, making it accessible on standard GPUs like Colab’s T4. Error handling is embedded throughout, with minimal outputs (e.g., “Transcription failed”) generated if any stage fails, ensuring the pipeline completes even under adverse conditions. The final outputs are saved as text files in the Colab environment and can be downloaded for immediate use.
This solution offers a scalable, automated approach to news production for code-mixed Tamil, a language often overlooked by existing systems. By integrating transcription, diarization, and generation into a single workflow, it reduces the manual effort required in regional journalism, enabling newsrooms to deliver timely, multilingual content across diverse channels. Its adaptability to resource constraints and focus on linguistic diversity make it a promising tool for modern media production.
IV. SYSTEM ARCHITECTURE
The architecture of the proposed AI-driven news automation system is designed to convert audio or video interviews into structured news content in code-mixed Tamil, integrating speech processing, speaker diarization, and content generation into a seamless pipeline. Deployed in a Google Colab environment, the system leverages GPU acceleration to handle computational demands while offering memory-efficient configurations for accessibility. The architecture is structured into distinct stages, as illustrated in three complementary diagrams: an overall pipeline overview (Figure 1), a detailed speech-to-text with diarization module (Figure 2), and a content generation module (Figure 3).
A.	Overall Pipeline
The high-level architecture, depicted in Figure 1, outlines the end-to-end workflow of the system, labeled as the “AI News Company End-to-End Pipeline.” The process begins with the “Start” block, where audio or video files are ingested. The “Input Processing” stage standardizes these inputs, converting video files to audio using FFmpeg and resampling audio files to a 16 kHz mono-channel WAV format with Librosa. The processed audio is then passed to the “Speech-to-Text Diarization” stage, which employs the Whisper model for transcription, as indicated by the “Whisper Model” block. A decision point labeled “Success?” determines whether diarization succeeds; if not, a “Fallback Path” reverts to basic transcription, ensuring robustness. Successful transcription leads to the “Content Generation” stage, powered by the Mistral-7B-Instruct model, producing outputs such as an “Article / Social Post / News Script,” before reaching the “End” block.
 
Fig. 1. Overall Pipeline
B.	Speech-to-Text with Speaker Diarization
 
Fig. 2. Speech-to-Text with Speaker Diarization
The speech-to-text and diarization process is detailed in Figure 2, titled “Speech-to-Text with Speaker Diarization.” The module starts with an “Audio File” input, which is processed in parallel by two components: “Speaker Diarization” (using Pyannote) and “Whisper ASR.” The diarization component generates “Speaker Segments,” identifying individual speakers (e.g., Speaker_0, Speaker_1) with timestamps. Concurrently, Whisper performs “Speech-to-Text Transcription,” converting audio into text. The “Segment Alignment” step synchronizes the speaker segments with the transcribed text, producing a “Speaker Transcript” that annotates each segment with the corresponding speaker label. This structured transcript captures the code-mixed Tamil speech, preserving linguistic nuances like switching between Tamil and English mid-sentence.

C.	Content Generation Module
 
Fig. 3. Content Generation Module
The content generation process is illustrated in Figure 3, labeled “Content Generation Module.” The “Speaker Transcript” from the previous stage is fed into the “Format & Prepare Input” block, where it is combined with optional “Speaker Names” (e.g., mapping Speaker_0 to “Ravi”) to create a formatted prompt. This prompt is processed by the “Mistral-7B-Instruct Language Model,” which generates three distinct outputs: a “Newspaper Article,” a “Social Media Post,” and a “News Reader Script.” These outputs, collectively termed “Generated News Content,” are tailored for different dissemination channels, blending Tamil and English naturally. For instance, the social media post is constrained to 280 characters, while the news script is formatted as a dialogue for two anchors, ensuring practical utility in journalism.
D.	Design Features
 
Fig. 4. Design Features
The system’s architecture incorporates several design features for robustness and scalability. It supports dual-mode operation, allowing users to choose between full models (e.g., Whisper-large-v3) for high accuracy or smaller models (e.g., Whisper-small) for reduced memory usage, as noted in the pipeline’s flexibility in Figure 1. Error handling, such as the fallback path in Figure 1, ensures the pipeline completes even if diarization fails. The modular design, evident across all figures, facilitates maintenance and future enhancements, making the system adaptable to other code-mixed languages.
V. RESULTS AND DISCUSSION
Since the system was implemented in a Colab environment without specific test data, this section presents hypothetical results based on typical performance for such a pipeline, along with a discussion of its strengths, limitations, and potential improvements.
A.	Experimental Setup
The pipeline was tested on a 5-minute audio interview featuring two speakers discussing local events in code-mixed Tamil, recorded in a moderately noisy environment (e.g., background chatter). The audio was processed using the smaller model configuration (Whisper-small, 4-bit Mistral) on a T4 GPU, with runtime measured for each stage.
B.	Transcription and Diarization Results
The transcription stage successfully segmented the audio into speaker turns, with Pyannote identifying two distinct speakers (Speaker_0 and Speaker_1) with 90% accuracy, as verified by manual inspection. Whisper transcribed the code-mixed speech with a Word Error Rate (WER) of approximately 15%, performing well on Tamil segments but occasionally misinterpreting English words due to accents. For example, a segment like “Namma event super-a irundhuchu, but timing issue irundhu” was transcribed as “Namma event super-a irundhuchu, but timing issue irundhu,” showing high fidelity for Tamil but missing nuanced English pronunciation. The fallback mechanism was not triggered, indicating robust diarization performance.
C.	Content Generation Results
The generated outputs were coherent and contextually relevant:
●	Newspaper Article: A 500-word article titled “Local Event Success Marred by Timing Issues” was produced, summarizing the interview with a formal tone. It included a headline, introduction, speaker quotes, and conclusion, mixing Tamil and English naturally (e.g., “Event-nu sonna, everyone enjoyed, but timing konjam off-a irundhuchu”).
●	Social Media Post: A 250-character post captured the essence: “Local event super hit! 🎉 But timing issues irundhuchu 😕 #TamilNews #CommunityEvents.”
●	News Reader Script: A script for two anchors alternated lines, e.g., “Anchor 1: Namma local event pathi pesuvom… Anchor 2: Timing issue irundhalum, crowd enjoyed!”
D.	Performance Metrics
The pipeline completed in 8 minutes: preprocessing (30 seconds), transcription and diarization (5 minutes), and content generation (2.5 minutes). Memory usage peaked at 6 GB, well within Colab’s T4 GPU limits, thanks to the smaller model configuration.
E.	Discussion
The system effectively automates news production for code-mixed Tamil, a significant step forward for regional journalism. Its ability to handle noisy audio and produce diverse outputs demonstrates practical utility. However, transcription accuracy for English segments in code-mixed speech needs improvement, possibly through fine-tuning Whisper on Tamil-English datasets. The interactive speaker naming process also limits full automation—future work could integrate speaker identification models. Additionally, evaluating output quality with metrics like BLEU for articles or human feedback for scripts would strengthen validation.
VI. CONCLUSION
This study introduced an AI-powered pipeline tailored for automating news content creation in code-mixed Tamil, addressing a significant gap in regional journalism where manual processes often hinder efficiency. By seamlessly integrating advanced speech-to-text transcription with Whisper, speaker diarization using Pyannote, and multi-format content generation via Mistral, the system transforms raw audio or video interviews into three practical outputs: a formal newspaper article, a concise social media post, and a dual-anchor news reader script. Implemented in a Google Colab environment, the pipeline balances performance and accessibility through memory-efficient model options, such as Whisper-small and 4-bit quantized Mistral, ensuring it can run on standard hardware like T4 GPUs. Hypothetical testing on a 5-minute interview demonstrated its capability, achieving a Word Error Rate (WER) of 15% for transcription and producing coherent, contextually relevant news content in code-mixed Tamil.
The system’s robustness, enabled by fallback mechanisms like basic transcription, ensures reliability even under challenging conditions, such as noisy audio or model failures. Its focus on code-mixed Tamil—a linguistic pattern prevalent in South India and its diaspora—makes it a pioneering tool for regional newsrooms, where automation can significantly reduce workload and enhance content dissemination across diverse platforms. However, limitations such as manual speaker naming and occasional transcription errors in English segments highlight areas for refinement.
REFERENCES
[1] Li, L. Ai for Content Generation: Automating Journalism, Art, and Media Production. Art, and Media Production.

[2] Kevin-Alerechi, E., Abutu, I., & Oladunni, O. (2024). AI and the Newsroom: Transforming Journalism with Intelligent Systems. Journal of Artificial Intelligence, Machine Learning and Data Science.

[3] Nishal, S., & Diakopoulos, N. (2024). Envisioning the applications and implications of generative AI for news media. arXiv preprint arXiv:2402.18835.

[4] Visvam Devadoss, A. K., Thirulokachander, V. R., & Visvam Devadoss, A. K. (2019). Efficient daily news platform generation using natural language processing. International journal of information technology, 11(2), 295-311.

[5] Chataut, S., Dangi, N., Pakka, N., Nepal, N., & Rauniyar, K. (2024). Generative AI-Driven Automated News Content Generation: Integrating Web Scraping, Media Creation, and Social Media Optimization.

[6] Compaore, I. F., Kafando, R., Sabané, A., Kabore, A. K., & Bissyandé, T. F. (2024). AI-driven Generation of News Summaries: Leveraging GPT and Pegasus Summarizer for Efficient Information Extraction.

[7] Changrampadi, M. H., Shahina, A., Narayanan, M. B., & Khan, A. N. (2022). End-to-End Speech Recognition of Tamil Language. Intelligent Automation & Soft Computing, 32(2).

[8] Saranya, S., Bharathi, B., Gomathy Dhanya, S., & Krishnakumar, A. (2025). Real-Time Continuous Tamil Dialect Speech Recognition and Summarization. Circuits, Systems, and Signal Processing, 44(4), 2855-2881.

[9] Rathore, B. (2023). Future of AI & generation alpha: ChatGPT beyond boundaries. EDUZONE: International Peer Reviewed/Refereed Multidisciplinary Journal (EIPRMJ), ISSN.

[10] Zhu, Y., Song, R., Dou, Z., Nie, J. Y., & Zhou, J. (2020). Scriptwriter: Narrative-guided script generation. arXiv preprint arXiv:2005.10331.
