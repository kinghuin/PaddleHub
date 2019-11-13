import paddlehub as hub
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import MODULE_HOME
import os
import shutil

url = "https://bj.bcebos.com/paddlehub/paddlehub_dev/roberta_wwm_ext_chinese_L-24_H-1024_A-16.hub_module-1.0.0-hub1.0.0-paddle1.5.0.tar.gz"

result, tips, module_zip_file = default_downloader.download_file(
    url=url,
    save_path=hub.CACHE_HOME,
    save_name="roberta_wwm_ext_chinese_L-24_H-1024_A-16",
    replace=True,
    print_progress=True)
result, tips, module_dir = default_downloader.uncompress(
    file=module_zip_file,
    dirname=MODULE_HOME,
    delete_file=True,
    print_progress=True)

save_path = os.path.join(MODULE_HOME,
                         "roberta_wwm_ext_chinese_L-24_H-1024_A-16")
if os.path.exists(save_path):
    shutil.rmtree(save_path)
shutil.move(module_dir, save_path)

dataset = hub.dataset.XNLI(language="zh")

dataset = hub.dataset.TNews()

# url="https://bj.bcebos.com/paddlehub/paddlehub_dev/lac.hub_module-1.1.0.tar.gz"
#
# result, tips, module_zip_file = default_downloader.download_file(
#     url=url,
#     save_path=hub.CACHE_HOME,
#     replace=True,
#     print_progress=True)
# result, tips, module_dir = default_downloader.uncompress(
#     file=module_zip_file,
#     dirname=MODULE_HOME,
#     delete_file=True,
#     print_progress=True)
#
# save_path = os.path.join(MODULE_HOME, "lac")
# if os.path.exists(save_path):
#     shutil.rmtree(save_path)
# shutil.move(module_dir, save_path)
