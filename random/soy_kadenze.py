import time, json, re, requests
from pathlib import Path
from random import randint
from playwright._impl._browser import Browser, Page
from playwright.sync_api import sync_playwright
from slugify import slugify
from typing import NamedTuple, List, Optional

'''
PyYAML==5.4.1
requests==2.25.1
python-slugify==4.0.1
playwright==1.9.1
'''

class Session(NamedTuple):
    course: str
    index: int
    name: str
    path: str


class Video(NamedTuple):
    session: Session
    index: int
    title: str
    url: str


class Settings(object):
    def __init__(self):
        #with open("configuration.yml", "r") as configfile:
        #    self.config = load(configfile, Loader=Loader)
        self.login = 'pwnagecorp2@gmail.com'  #self.config["kadenze"]["login"]
        self.password = 'yolotrap234' #self.config["kadenze"]["password"]
        self.path = r'C:\Users\i_hat\Desktop\bastl\kadenze'  #self.config["download"]["path"]
        self.courses = ["creative-applications-of-deep-learning-with-tensorflow-i"] #self.config["download"]["courses"]
        self.video_format = '720' #self.config["download"]["resolution"]
        self.videos_titles = True #self.config["download"]["videos_titles"]
        self.selected_only = True

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(Settings, cls).__new__(cls)
        return cls.instance


def format_course(course: str) -> str:
    formatted_course = course.split("/")[-1]
    return f"{formatted_course}"






def get_courses_from_json(response: str) -> List[str]:
    try:
        json_string = json.loads(response)
        courses = [course["course_path"] for course in json_string["courses"]]
    except ValueError:
        print("Error getting the courses list. Check that you're enrolled on selected courses.")
        courses = []
    return courses


def get_sessions_from_json(response: str, course: str) -> List[Session]:
    sessions = []
    try:
        d = json.loads(response)
        lectures = d["lectures"]
        for i, lecture in enumerate(lectures, start=1):
            try:
                session = Session(course, lecture["order"], slugify(lecture["title"]), lecture["course_session_path"])
                sessions.append(session)
            except Exception as e:
                print(f"Error while extracting session metadata from course {course} at index {i}: {e}")
    except Exception as e:
        print(f"Error while extracting session metadata from course {course}: {e}")
    return sessions


def get_videos_from_json(response: str, resolution: int, session: Session) -> List[Video]:
    videos = []
    try:
        d = json.loads(response)
        video_format = f"h264_{resolution}_url"
        vs = d["videos"]
        for i, v in enumerate(vs, start=1):
            try:
                video = Video(session, v["order"], v["title"], v[video_format])
                videos.append(video)
            except Exception as e:
                print(f"Error while extracting video metadata from session {session.name} at index {i}: {e}")
    except Exception as e:
        print(f"Error getting videos: {e}")
    return videos


def get_video_title(video_title: str, filename: str) -> str:
    try:
        slug = slugify(video_title)
        video_title = "_".join(filename.split(".")[:-1]) + "p_" + slug + "." + filename.split(".")[-1]
    except IndexError:
        video_title = filename
    return video_title


def write_video(video_url: str, full_path: str, filename: str, chunk_size: int = 4096):
    try:
        size = int(requests.head(video_url).headers["Content-Length"])
        size_on_disk = check_if_file_exists(full_path, filename)
        if size_on_disk < size:
            fd = Path(full_path)
            fd.mkdir(parents=True, exist_ok=True)
            with open(fd / filename, "wb") as f:
                r = requests.get(video_url, stream=True)
                current_size = 0
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    current_size += chunk_size
                    s = progress(current_size, size, filename)
                    print(s, end="", flush=True)
                print(s)
        else:
            print(f"{filename} already downloaded, skipping...")
    except Exception as e:
        print(f"Error while writing video to {full_path}/{filename}: {e}")


def check_if_file_exists(full_path: str, filename: str) -> int:
    f = Path(full_path + "/" + filename)
    if f.exists():
        return f.stat().st_size
    else:
        return 0


def progress(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    s = "[%s] %s%s filename: %s\r" % (bar, percents, "%", status)
    return s


def execute_login(browser: Browser) -> Page:
    print("Signing in www.kadenze.com ...")
    page: Page = browser.new_page()
    page.goto(BASE_URL)
    page.mouse.click(0, 0)
    page.click("#email-login-btn")
    page.fill("input#login_user_email", conf.login)
    page.fill("input#login_user_password", conf.password)
    time.sleep(randint(3, 8))
    page.click("//*[@id='login_user']/button")
    time.sleep(randint(3, 8))
    return page


def list_courses(page: Page) -> List[str]:
    try:
        page.goto(BASE_URL + "/my_courses")
        page.click("text=View all")
        time.sleep(randint(3, 8))
        div_courses = page.query_selector("div#my_courses")
        json_courses = div_courses.get_attribute("data-courses-data")
        courses = get_courses_from_json(json_courses)
    except Exception as e:
        print(f"Error while listing courses: {e}")
        courses = []
    return courses


def extract_sessions(page: Page, course: str) -> List[Session]:
    sessions = []
    print(f"Parsing course: {course}")
    sessions_url = "/".join((BASE_URL, "courses", course, "sessions"))
    try:
        page.goto(sessions_url)
        div_sessions: Page = page.query_selector("div#lectures_json")
        json_sessions = div_sessions.get_attribute("data-lectures-json")
        sessions = get_sessions_from_json(json_sessions, course)
    except Exception as e:
        print(f"Error while extracting sessions from course {course}: {e}")
    return sessions


def extract_session_videos(page: Page, session: Session) -> List[Video]:
    videos = []
    try:
        print(f"Parsing session: {session.name}")
        page.goto(BASE_URL + session.path)
        div_videos: Page = page.query_selector("#video_json")
        json_videos = div_videos.get_attribute("value")
        videos = get_videos_from_json(json_videos, conf.video_format, session)
    except Exception as e:
        print(f"Error while extracting videos from session {session.name}: {e}")
    return videos



filename_pattern = re.compile("file/(.*\.mp4)\?")
BASE_URL = "https://www.kadenze.com"
conf = Settings()
p = sync_playwright().start()
browser = p.firefox.launch(headless=True)


page = execute_login(browser)
enrolled_courses = [format_course(course) for course in list_courses(page)]
courses = [c for c in enrolled_courses if any(substring in c for substring in conf.courses)]

courses

course = courses[0]


sessions = extract_sessions(page, course)
_videos = [extract_session_videos(page, session) for session in sessions]



session = sessions[1]
page.goto(BASE_URL + session.path)
div_videos: Page = page.query_selector("#video_json")
json_videos = div_videos.get_attribute("value")
videos = get_videos_from_json(json_videos, conf.video_format, session)
div_videos
json_videos


ff = open(r'C:\Users\i_hat\Desktop\soy.txt', 'w')
ff.write(json_videos)
ff.close()


videos = _videos[1][15:16]
#videos = [v for sublist in videos for v in sublist]
videos

for video in videos:
    try:
        try:
            filename = re.search(filename_pattern, video.url).group(1)
        except Exception:
            filename = None

        if filename is not None:
            session_prefix = str(video.session.index) + "-" + video.session.name
            full_path = conf.path + "/" + video.session.course[:10] + "/" + session_prefix[:10]
            Path(full_path).mkdir(parents=True, exist_ok=True)
            if conf.videos_titles:
                filename = get_video_title(video.title, filename)
            write_video(video.url, full_path, filename)
        else:
            print(f"Could not extract filename of video {video.title} from session {video.session.name} and course {video.session.course}, skipping...")
    except Exception as e:
        print(f"Error while downloading video {video.title} from course {course}: {e}")


page.close()
browser.close()
p.stop()



