import argparse
import csv
import json
import os
import tempfile
import time

import openai

# Usage examples:
#   py test-endpoints.py
#   py test-endpoints.py --only responses,audio,files
#   py test-endpoints.py --skip videos,evals,fine_tuning
#   set OPENAI_RUN_EXPENSIVE=1; py test-endpoints.py --only videos,batches
#   set OPENAI_REALTIME=1; py test-endpoints.py --only realtime
#   set OPENAI_RUN_EXPENSIVE=1; set OPENAI_REALTIME=1; set OPENAI_FORCE_CALLS=1; py test-endpoints.py

# Initialize the OpenAI client once so all test functions can reuse it.
# NOTE: For safety, API keys should be loaded from environment variables.
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
client = openai.OpenAI(api_key=API_KEY)

RESULTS = {"OK": 0, "ERROR": 0, "SKIPPED": 0, "STATEFUL": 0, "QUEUED": 0, "WARN": 0}
RESULT_ROWS = []
NOTE_WRAP = 120
SDP_SAMPLE = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=codex\r\nt=0 0\r\nm=audio 9 RTP/AVP 0\r\nc=IN IP4 0.0.0.0\r\na=rtpmap:0 PCMU/8000\r\n"
REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def raw_request(method, path, json_body=None, files=None, params=None):
    http = getattr(client, "_client", None)
    if http is None:
        raise RuntimeError("Low-level client not available on this SDK")
    if hasattr(http, "request"):
        return http.request(method, path, json=json_body, files=files, params=params)
    raise RuntimeError("Low-level request method not available on this SDK")

def log_result(group, status, note):
    RESULTS[status] = RESULTS.get(status, 0) + 1
    note = "" if note is None else str(note)
    RESULT_ROWS.append((group, status, note))
    chunks = [note[i : i + NOTE_WRAP] for i in range(0, len(note), NOTE_WRAP)] or [""]
    print(f"{group:<25} | {status:<10} | {chunks[0]}")
    for chunk in chunks[1:]:
        print(f"{'':<25} | {'':<10} | {chunk}")


def safe_call(group, fn, ok_note="OK", skip_note="Not supported by SDK"):
    try:
        result = fn()
        log_result(group, "OK", ok_note)
        return result
    except AttributeError:
        log_result(group, "SKIPPED", skip_note)
        return None
    except Exception as e:
        log_result(group, "ERROR", str(e))
        return None


def poll_until(get_fn, done_pred, timeout_s, interval_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        obj = get_fn()
        if done_pred(obj):
            return obj
        time.sleep(interval_s)
    return None


def best_effort_cleanup(label, fn, strict=False):
    try:
        fn()
        log_result(label, "OK", "Cleanup succeeded")
        return True
    except Exception as e:
        if strict:
            raise
        log_result(label, "WARN", f"Cleanup failed: {str(e)}")
        return False


def safe_remove(path, label="Cleanup Remove"):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        log_result(label, "WARN", f"Remove failed: {str(e)}")


def get_attr(obj, path):
    cur = obj
    for name in path:
        cur = getattr(cur, name, None)
        if cur is None:
            return None
    return cur


def should_run(args, suite_key):
    if args.only and suite_key not in args.only:
        return False
    if args.skip and suite_key in args.skip:
        return False
    return True


def audit_all_endpoints(args):
    print("Starting Full OpenAI Endpoint Audit")
    print(f"{'-'*90}")
    print(f"{'Endpoint Group':<25} | {'Status':<10} | {'Note'}")
    print(f"{'-'*90}")

    suites = [
        ("chat", test_chat),
        ("responses", test_responses),
        ("responses_stream", test_responses_stream),
        ("conversations", test_conversations),
        ("audio", test_audio),
        ("images", test_images),
        ("videos", test_videos),
        ("embeddings", test_embeddings),
        ("moderations", test_moderations),
        ("files", test_files),
        ("uploads", test_uploads),
        ("batches", test_batches),
        ("evals", test_evals),
        ("vector_stores", test_vector_stores),
        ("chatkit", test_chatkit),
        ("containers", test_containers),
        ("skills", test_skills),
        ("realtime", test_realtime),
        ("legacy_assistants", test_legacy_assistants),
    ]

    for key, fn in suites:
        if should_run(args, key):
            fn(args)
        else:
            log_result(key, "SKIPPED", "Skipped by filter")

    print(f"{'-'*90}")
    print(
        "Summary: "
        f"OK={RESULTS.get('OK', 0)} "
        f"ERROR={RESULTS.get('ERROR', 0)} "
        f"SKIPPED={RESULTS.get('SKIPPED', 0)} "
        f"STATEFUL={RESULTS.get('STATEFUL', 0)} "
        f"QUEUED={RESULTS.get('QUEUED', 0)} "
        f"WARN={RESULTS.get('WARN', 0)}"
    )
    write_csv("test-results.csv")


def write_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Endpoint", "Result", "Note"])
        writer.writerows(RESULT_ROWS)


def test_chat(args):
    def do_call():
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            store=True,
        )

    resp = safe_call("Chat Completions", do_call, "Chat completion created")
    if resp:
        fp = getattr(resp, "system_fingerprint", "unknown")
        log_result("Chat Completions", "OK", f"Fingerprint: {fp}")


def test_responses(args):
    def create_simple():
        return client.responses.create(
            model="gpt-5.2",
            input="ZDR Test",
        )

    resp = safe_call("Responses Create", create_simple, "Response created")
    resp_id = getattr(resp, "id", None) if resp else None

    if resp_id:
        retrieve = get_attr(client, ["responses", "retrieve"])
        if retrieve:
            safe_call("Responses Retrieve", lambda: retrieve(resp_id), "Response retrieved")
        delete = get_attr(client, ["responses", "delete"])
        if delete:
            safe_call("Responses Delete", lambda: delete(resp_id), "Response deleted")

    background = get_attr(client, ["responses", "create"])
    if background:
        try:
            client.responses.create(model="gpt-5.2", input="ZDR Test", background=True)
            log_result("Responses Background", "STATEFUL", "Background mode active")
        except Exception as e:
            log_result("Responses Background", "ERROR", str(e))

    try:
        client.responses.create(
            model="gpt-5.2",
            tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
            input="Calculate 1+1",
            store=False,
        )
        log_result("Code Interpreter", "STATEFUL", "Tool executed")
    except Exception as e:
        log_result("Code Interpreter", "ERROR", str(e))

    input_items = get_attr(client, ["responses", "input_items"])
    if input_items:
        if getattr(input_items, "list", None) and resp_id:
            safe_call(
                "Responses Input Items",
                lambda: input_items.list(resp_id),
                "Input items listed",
            )
        if getattr(input_items, "count_tokens", None) and resp_id:
            safe_call(
                "Responses Count Tokens",
                lambda: input_items.count_tokens(resp_id),
                "Tokens counted",
            )

    cancel = get_attr(client, ["responses", "cancel"])
    if cancel and resp_id:
        safe_call("Responses Cancel", lambda: cancel(resp_id), "Response cancelled")

    compact = get_attr(client, ["responses", "compact"])
    if compact and resp_id:
        safe_call("Responses Compact", lambda: compact(resp_id), "Response compacted")


def test_responses_stream(args):
    stream_method = get_attr(client, ["responses", "stream"])
    if not stream_method:
        log_result("Responses Stream", "SKIPPED", "Streaming not supported")
        return

    try:
        with stream_method(model="gpt-5.2", input="Stream a short reply") as stream:
            for _ in stream:
                break
        log_result("Responses Stream", "OK", "Received stream event")
    except Exception as e:
        log_result("Responses Stream", "ERROR", str(e))


def test_conversations(args):
    conv_create = get_attr(client, ["conversations", "create"])
    if not conv_create:
        log_result("Conversations", "SKIPPED", "Conversations not supported")
        return

    conv = safe_call("Conversations Create", lambda: conv_create(), "Conversation created")
    conv_id = getattr(conv, "id", None) if conv else None

    if conv_id:
        conv_get = get_attr(client, ["conversations", "retrieve"])
        if conv_get:
            safe_call("Conversations Retrieve", lambda: conv_get(conv_id), "Retrieved")
        conv_update = get_attr(client, ["conversations", "update"])
        if conv_update:
            safe_call("Conversations Update", lambda: conv_update(conv_id, metadata={"k": "v"}), "Updated")
        conv_delete = get_attr(client, ["conversations", "delete"])
        if conv_delete:
            safe_call("Conversations Delete", lambda: conv_delete(conv_id), "Deleted")

    items = get_attr(client, ["items"])
    if items and getattr(items, "create", None):
        item = safe_call(
            "Items Create",
            lambda: items.create(conversation_id=conv_id, content="hello"),
            "Item created",
        )
        item_id = getattr(item, "id", None) if item else None
        if item_id and getattr(items, "retrieve", None):
            safe_call("Items Retrieve", lambda: items.retrieve(item_id), "Item retrieved")
        if getattr(items, "list", None) and conv_id:
            safe_call("Items List", lambda: items.list(conversation_id=conv_id), "Items listed")
        if item_id and getattr(items, "delete", None):
            safe_call("Items Delete", lambda: items.delete(item_id), "Item deleted")
    else:
        if env_bool("OPENAI_FORCE_CALLS", False) and conv_id:
            try:
                raw_request(
                    "POST",
                    "/v1/items",
                    json_body={"conversation_id": conv_id, "content": "hello"},
                )
                log_result("Items Create", "OK", "Raw /v1/items call succeeded")
            except Exception as e:
                log_result("Items Create", "ERROR", str(e))
        else:
            log_result("Items", "SKIPPED", "Items not supported")


def test_audio(args):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input="This is a short audio test.",
        ) as response:
            response.stream_to_file(tmp_path)

        with open(tmp_path, "rb") as audio_file:
            client.audio.transcriptions.create(model="gpt-4o-transcribe", file=audio_file)

        log_result("Audio (In/Out)", "OK", "TTS + transcription succeeded")
    except Exception as e:
        log_result("Audio (In/Out)", "ERROR", str(e))
    finally:
        safe_remove(tmp_path, "Audio Cleanup")

    if tmp_path and os.path.exists(tmp_path):
        try:
            with open(tmp_path, "rb") as audio_file:
                client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                )
            log_result("Audio Translation", "OK", "Translation created")
        except Exception as e:
            log_result("Audio Translation", "ERROR", str(e))

    voices = get_attr(client, ["audio", "voices"])
    if voices and getattr(voices, "create", None):
        voice = safe_call(
            "Voice Create",
            lambda: voices.create(name="audit-voice", description="audit voice"),
            "Voice created",
        )
        voice_id = getattr(voice, "id", None) if voice else None
        consents = get_attr(client, ["audio", "voice_consents"])
        if consents and getattr(consents, "list", None):
            safe_call("Voice Consents List", lambda: consents.list(voice_id=voice_id), "Consents listed")
        if voice_id and getattr(voices, "delete", None):
            safe_call("Voice Delete", lambda: voices.delete(voice_id), "Voice deleted")
    else:
        if env_bool("OPENAI_FORCE_CALLS", False):
            try:
                raw_request(
                    "POST",
                    "/v1/audio/voices",
                    json_body={"name": "audit-voice", "description": "audit voice"},
                )
                log_result("Voice Create", "OK", "Raw /v1/audio/voices call succeeded")
            except Exception as e:
                log_result("Voice Create", "ERROR", str(e))
        else:
            log_result("Voice Create", "SKIPPED", "Voice APIs not supported")


def test_images(args):
    try:
        client.responses.create(
            model="gpt-5",
            input="Generate a simple blue square on a white background.",
            tools=[{"type": "image_generation"}],
        )
        log_result("Images Generate", "OK", "Image generated")
    except Exception as e:
        log_result("Images Generate", "ERROR", str(e))

    images_api = get_attr(client, ["images"])
    if images_api and getattr(images_api, "edits", None):
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            images_api.edits(
                model="gpt-image-1",
                image=open(tmp_path, "rb"),
                prompt="Add a red dot",
            )
            log_result("Images Edit", "OK", "Image edited")
        except Exception as e:
            log_result("Images Edit", "ERROR", str(e))
        finally:
            safe_remove(tmp_path, "Images Edit Cleanup")
    else:
        if env_bool("OPENAI_FORCE_CALLS", False):
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                with open(tmp_path, "rb") as img_file:
                    raw_request(
                        "POST",
                        "/v1/images/edits",
                        files={"image": img_file},
                        params={"model": "gpt-image-1", "prompt": "Add a red dot"},
                    )
                log_result("Images Edit", "OK", "Raw /v1/images/edits call succeeded")
            except Exception as e:
                log_result("Images Edit", "ERROR", str(e))
            finally:
                safe_remove(tmp_path, "Images Edit Cleanup")
        else:
            log_result("Images Edit", "SKIPPED", "Image edit not supported")


def test_videos(args):
    if not env_bool("OPENAI_RUN_EXPENSIVE", False):
        log_result("Videos Create", "SKIPPED", "OPENAI_RUN_EXPENSIVE=0")
        return

    create = get_attr(client, ["videos", "create"])
    if not create:
        log_result("Videos Create", "SKIPPED", "Videos not supported")
        return

    job = safe_call(
        "Videos Create",
        lambda: create(
            model="sora-2",
            prompt="A short test clip of a red ball rolling across a table.",
            seconds="4",
            size="720x1280",
        ),
        "Video job created",
    )
    job_id = getattr(job, "id", None) if job else None

    retrieve = get_attr(client, ["videos", "retrieve"])
    if retrieve and job_id:
        safe_call("Videos Retrieve", lambda: retrieve(job_id), "Video retrieved")

    list_fn = get_attr(client, ["videos", "list"])
    if list_fn:
        safe_call("Videos List", list_fn, "Video list retrieved")

    cancel = get_attr(client, ["videos", "cancel"])
    if cancel and job_id:
        safe_call("Videos Cancel", lambda: cancel(job_id), "Video cancelled")

    delete = get_attr(client, ["videos", "delete"])
    if delete and job_id:
        safe_call("Videos Delete", lambda: delete(job_id), "Video deleted")


def test_embeddings(args):
    safe_call(
        "Embeddings",
        lambda: client.embeddings.create(model="text-embedding-3-small", input="test"),
        "Embedding created",
    )


def test_moderations(args):
    safe_call(
        "Moderations",
        lambda: client.moderations.create(model="omni-moderation-latest", input="test"),
        "Moderation created",
    )


def test_files(args):
    tmp_path = None
    uploaded = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write("hello")
            tmp_path = tmp_file.name

        uploaded = client.files.create(file=open(tmp_path, "rb"), purpose="assistants")
        log_result("Files Create", "OK", f"File created: {uploaded.id}")
    except Exception as e:
        log_result("Files Create", "ERROR", str(e))

    if uploaded:
        file_id = uploaded.id
        safe_call("Files Retrieve", lambda: client.files.retrieve(file_id), "File retrieved")
        safe_call("Files Content", lambda: client.files.content(file_id), "File content retrieved")
        safe_call("Files List", lambda: client.files.list(), "Files listed")
        safe_call("Files Delete", lambda: client.files.delete(file_id), "File deleted")

    safe_remove(tmp_path, "Files Cleanup")


def test_uploads(args):
    uploads = get_attr(client, ["uploads"])
    if not uploads:
        log_result("Uploads", "SKIPPED", "Uploads not supported")
        return

    tmp_path = None
    upload_id = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write("upload test")
            tmp_path = tmp_file.name

        bytes_len = os.path.getsize(tmp_path)
        upload = uploads.create(
            filename="upload.txt",
            purpose="assistants",
            bytes=bytes_len,
            mime_type="text/plain",
        )
        upload_id = getattr(upload, "id", None)
        with open(tmp_path, "rb") as upload_file:
            part = uploads.parts.create(
                upload_id=upload_id,
                data=upload_file,
            )
        uploads.complete(upload_id=upload_id, part_ids=[part.id])
        log_result("Uploads Complete", "OK", f"Upload completed: {upload_id}")
    except Exception as e:
        log_result("Uploads", "ERROR", str(e))
    finally:
        if upload_id and getattr(uploads, "cancel", None):
            best_effort_cleanup("Uploads Cancel", lambda: uploads.cancel(upload_id))
        safe_remove(tmp_path, "Uploads Cleanup")


def test_batches(args):
    if not env_bool("OPENAI_RUN_EXPENSIVE", False):
        log_result("Batches Create", "SKIPPED", "OPENAI_RUN_EXPENSIVE=0")
        return

    tmp_path = None
    try:
        request_line = {
            "custom_id": "audit-1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": "gpt-4o-mini", "input": "ping"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(json.dumps(request_line) + "\n")
            tmp_path = tmp_file.name

        with open(tmp_path, "rb") as batch_file:
            uploaded = client.files.create(file=batch_file, purpose="batch")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        log_result("Batches Create", "STATEFUL", f"Batch created: {batch.id}")
        safe_call("Batches Retrieve", lambda: client.batches.retrieve(batch.id), "Batch retrieved")
        safe_call("Batches List", lambda: client.batches.list(), "Batches listed")
        safe_call("Batches Cancel", lambda: client.batches.cancel(batch.id), "Batch cancelled")
    except Exception as e:
        log_result("Batches Create", "ERROR", str(e))
    finally:
        safe_remove(tmp_path, "Batches Cleanup")


def test_evals(args):
    if not env_bool("OPENAI_RUN_EXPENSIVE", False):
        log_result("Evals Create", "SKIPPED", "OPENAI_RUN_EXPENSIVE=0")
        return

    evals = get_attr(client, ["evals"])
    if not evals:
        log_result("Evals", "SKIPPED", "Evals not supported")
        return

    created = safe_call(
        "Evals Create",
        lambda: evals.create(
            name="audit-eval",
            data_source_config={
                "type": "custom",
                "item_schema": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}, "expected": {"type": "string"}},
                    "required": ["prompt", "expected"],
                },
                "include_sample_schema": True,
            },
            testing_criteria=[
                {
                    "type": "string_check",
                    "name": "Match output to expected",
                    "input": "{{ sample.output_text }}",
                    "operation": "eq",
                    "reference": "{{ item.expected }}",
                }
            ],
        ),
        "Eval created",
    )
    eval_id = getattr(created, "id", None) if created else None
    if eval_id:
        safe_call("Evals Retrieve", lambda: evals.retrieve(eval_id), "Eval retrieved")
        safe_call("Evals List", lambda: evals.list(), "Evals listed")
        safe_call("Evals Delete", lambda: evals.delete(eval_id), "Eval deleted")

    runs = get_attr(evals, ["runs"])
    if runs and getattr(runs, "create", None) and eval_id:
        run = safe_call(
            "Eval Run Create",
            lambda: runs.create(
                eval_id=eval_id,
                data_source={
                    "type": "responses",
                    "source": {"type": "inline"},
                    "items": [{"input": "1+1?", "expected": "2"}],
                },
            ),
            "Run created",
        )
        run_id = getattr(run, "id", None) if run else None
        if run_id:
            safe_call("Eval Run Retrieve", lambda: runs.retrieve(run_id), "Run retrieved")
            safe_call("Eval Run List", lambda: runs.list(eval_id=eval_id), "Runs listed")
            safe_call("Eval Run Cancel", lambda: runs.cancel(run_id), "Run cancelled")



def test_vector_stores(args):
    vs = get_attr(client, ["vector_stores"])
    if not vs:
        log_result("Vector Stores", "SKIPPED", "Vector stores not supported")
        return

    store = safe_call("Vector Store Create", lambda: vs.create(name="audit-store"), "Store created")
    store_id = getattr(store, "id", None) if store else None
    if store_id:
        safe_call("Vector Store Retrieve", lambda: vs.retrieve(store_id), "Store retrieved")
        safe_call("Vector Store List", lambda: vs.list(), "Stores listed")
        safe_call("Vector Store Search", lambda: vs.search(store_id, query="test"), "Search executed")

    files_api = get_attr(vs, ["files"])
    if files_api and store_id:
        file_obj = safe_call(
            "Vector Store File Create",
            lambda: files_api.create(
                vector_store_id=store_id,
                file_id=client.files.create(
                    file=open(__file__, "rb"),
                    purpose="assistants",
                ).id,
            ),
            "Vector store file created",
        )
        file_id = getattr(file_obj, "id", None) if file_obj else None
        if file_id:
            safe_call(
                "Vector Store File Retrieve",
                lambda: files_api.retrieve(vector_store_id=store_id, file_id=file_id),
                "File retrieved",
            )
            safe_call(
                "Vector Store File List",
                lambda: files_api.list(vector_store_id=store_id),
                "Files listed",
            )
            safe_call(
                "Vector Store File Delete",
                lambda: files_api.delete(vector_store_id=store_id, file_id=file_id),
                "File deleted",
            )

    if store_id:
        safe_call("Vector Store Delete", lambda: vs.delete(store_id), "Store deleted")


def test_chatkit(args):
    chatkit = get_attr(client, ["chatkit"])
    if not chatkit:
        log_result("ChatKit", "SKIPPED", "ChatKit not supported")
        return

    sessions = get_attr(chatkit, ["sessions"])
    if sessions and getattr(sessions, "create", None):
        session = safe_call("ChatKit Session Create", lambda: sessions.create(), "Session created")
        session_id = getattr(session, "id", None) if session else None
        if session_id and getattr(sessions, "retrieve", None):
            safe_call("ChatKit Session Retrieve", lambda: sessions.retrieve(session_id), "Session retrieved")
        if getattr(sessions, "list", None):
            safe_call("ChatKit Session List", lambda: sessions.list(), "Sessions listed")
        if session_id and getattr(sessions, "delete", None):
            safe_call("ChatKit Session Delete", lambda: sessions.delete(session_id), "Session deleted")

    threads = get_attr(chatkit, ["threads"])
    if threads and getattr(threads, "create", None):
        thread = safe_call("ChatKit Thread Create", lambda: threads.create(), "Thread created")
        thread_id = getattr(thread, "id", None) if thread else None
        if thread_id and getattr(threads, "retrieve", None):
            safe_call("ChatKit Thread Retrieve", lambda: threads.retrieve(thread_id), "Thread retrieved")
        if getattr(threads, "list", None):
            safe_call("ChatKit Thread List", lambda: threads.list(), "Threads listed")
        if thread_id and getattr(threads, "delete", None):
            safe_call("ChatKit Thread Delete", lambda: threads.delete(thread_id), "Thread deleted")

    items = get_attr(chatkit, ["items"])
    if items and getattr(items, "create", None):
        item = safe_call("ChatKit Item Create", lambda: items.create(content="test"), "Item created")
        item_id = getattr(item, "id", None) if item else None
        if item_id and getattr(items, "retrieve", None):
            safe_call("ChatKit Item Retrieve", lambda: items.retrieve(item_id), "Item retrieved")
        if getattr(items, "list", None):
            safe_call("ChatKit Item List", lambda: items.list(), "Items listed")
        if item_id and getattr(items, "delete", None):
            safe_call("ChatKit Item Delete", lambda: items.delete(item_id), "Item deleted")


def test_containers(args):
    containers = get_attr(client, ["containers"])
    if not containers:
        log_result("Containers", "SKIPPED", "Containers not supported")
        return

    container = safe_call("Containers Create", lambda: containers.create(name="audit-container"), "Container created")
    container_id = getattr(container, "id", None) if container else None
    if container_id:
        safe_call("Containers Retrieve", lambda: containers.retrieve(container_id), "Container retrieved")
        safe_call("Containers List", lambda: containers.list(), "Containers listed")
        safe_call("Containers Delete", lambda: containers.delete(container_id), "Container deleted")

    files_api = get_attr(containers, ["files"])
    if files_api and container_id:
        file_obj = safe_call(
            "Container File Create",
            lambda: files_api.create(container_id=container_id, file=open(__file__, "rb")),
            "Container file created",
        )
        file_id = getattr(file_obj, "id", None) if file_obj else None
        if file_id:
            safe_call(
                "Container File Retrieve",
                lambda: files_api.retrieve(container_id=container_id, file_id=file_id),
                "File retrieved",
            )
            safe_call(
                "Container File List",
                lambda: files_api.list(container_id=container_id),
                "Files listed",
            )
            safe_call(
                "Container File Delete",
                lambda: files_api.delete(container_id=container_id, file_id=file_id),
                "File deleted",
            )


def test_skills(args):
    skills = get_attr(client, ["skills"])
    if not skills:
        log_result("Skills", "SKIPPED", "Skills not supported")
        return

    skill = safe_call("Skills Create", lambda: skills.create(name="audit-skill"), "Skill created")
    skill_id = getattr(skill, "id", None) if skill else None
    if skill_id:
        safe_call("Skills Retrieve", lambda: skills.retrieve(skill_id), "Skill retrieved")
        safe_call("Skills List", lambda: skills.list(), "Skills listed")
        safe_call("Skills Delete", lambda: skills.delete(skill_id), "Skill deleted")

    versions = get_attr(skills, ["versions"])
    if versions and getattr(versions, "create", None) and skill_id:
        version = safe_call(
            "Skill Version Create",
            lambda: versions.create(skill_id=skill_id, content="version"),
            "Version created",
        )
        version_id = getattr(version, "id", None) if version else None
        if version_id:
            safe_call(
                "Skill Version Retrieve",
                lambda: versions.retrieve(skill_id=skill_id, version_id=version_id),
                "Version retrieved",
            )
            safe_call(
                "Skill Version List",
                lambda: versions.list(skill_id=skill_id),
                "Versions listed",
            )
            safe_call(
                "Skill Version Delete",
                lambda: versions.delete(skill_id=skill_id, version_id=version_id),
                "Version deleted",
            )


def test_realtime(args):
    if not env_bool("OPENAI_REALTIME", False):
        log_result("Realtime", "SKIPPED", "OPENAI_REALTIME=0")
        return

    realtime = get_attr(client, ["realtime"])
    if not realtime:
        log_result("Realtime", "SKIPPED", "Realtime not supported")
        return

    client_secrets = get_attr(realtime, ["client_secrets"])
    if client_secrets and getattr(client_secrets, "create", None):
        safe_call(
            "Realtime Client Secret",
            lambda: client_secrets.create(model=REALTIME_MODEL),
            "Client secret created",
        )

    calls = get_attr(realtime, ["calls"])
    if calls and getattr(calls, "create", None):
        call = safe_call(
            "Realtime Call Create",
            lambda: calls.create(sdp=SDP_SAMPLE),
            "Call created",
        )
        call_id = getattr(call, "id", None) if call else None
        if call_id and getattr(calls, "accept", None):
            safe_call("Realtime Call Accept", lambda: calls.accept(call_id), "Call accepted")
        if call_id and getattr(calls, "hangup", None):
            safe_call("Realtime Call Hangup", lambda: calls.hangup(call_id), "Call hung up")


def test_legacy_assistants(args):
    try:
        client.beta.threads.create()
        log_result("Legacy Threads", "STATEFUL", "Database persistence is active")
    except Exception as e:
        log_result("Legacy Threads", "ERROR", str(e))


def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI API endpoint audit")
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--skip", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    args.only = {s.strip() for s in args.only.split(",") if s.strip()}
    args.skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    audit_all_endpoints(args)


if __name__ == "__main__":
    main()

