import pytest
import shutil
from pathlib import Path
from src.utils.file_manager import FileManager
from src.utils.audio import is_supported_format, get_audio_info

# --- FileManager Tests ---

@pytest.fixture
def temp_dir():
    path = Path("tests/temp_test")
    path.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        shutil.rmtree(path)

def test_file_manager_create_temp_dir(temp_dir):
    fm = FileManager(temp_dir=temp_dir)
    user_id = 12345
    user_path = fm.get_user_dir(user_id)
    assert user_path.exists()
    # FileManager creates temp_dir/audio/user_id structure
    assert str(user_id) in str(user_path)
    assert "audio" in str(user_path)

def test_file_manager_cleanup(temp_dir):
    fm = FileManager(temp_dir=temp_dir)
    user_id = 12345
    user_path = fm.get_user_dir(user_id)
    
    # Create a dummy file
    (user_path / "test.txt").touch()
    assert (user_path / "test.txt").exists()
    
    # Use cleanup_user_files instead of cleanup_user_dir
    fm.cleanup_user_files(user_id)
    # File should be gone
    assert not (user_path / "test.txt").exists()

# --- Audio Tests ---

def test_audio_utils_is_supported_format():
    assert is_supported_format(Path("audio.mp3"))
    assert is_supported_format(Path("voice.ogg"))
    assert is_supported_format(Path("recording.wav"))
    assert not is_supported_format(Path("image.jpg"))
    assert not is_supported_format(Path("document.pdf"))

@pytest.mark.asyncio
async def test_audio_utils_get_duration(mocker):
    # Mock asyncio.create_subprocess_exec
    mock_process = mocker.AsyncMock()
    mock_process.communicate.return_value = (
        b'{"format": {"duration": "120.5"}, "streams": [{"codec_type": "audio", "sample_rate": "44100", "channels": 2}]}',
        b""
    )
    mock_process.returncode = 0
    
    mocker.patch("asyncio.create_subprocess_exec", return_value=mock_process)
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.stat", return_value=mocker.Mock(st_size=1000))
    
    info = await get_audio_info(Path("dummy_path.mp3"))
    assert info.duration_seconds == 120.5
    assert info.is_valid is True
