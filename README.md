# paddlepaddleAI
百度 PaddlePaddle AI 大赛——综艺节目精彩片段预测

## 安装PaddlePaddle（linux服务器）
最坑的还是环境问题，paddle目前只支持python2的环境，所有安装需要使用pip2.7进行安装
```
pip2.7 install paddlepaddle
```
但是会出现权限不够的情况，所以需要加上user属性
```
pip2.7 install paddlepaddle-gpu --user
```
另外在导入paddle.v2时，也需要系统环境的支持，之前使用虚拟环境根本就无法成功导入，而且同样也是需要使用python2.7进行执行，这样每次都是需要打python2.7的，所以可以使用命令重命名，alias
```
alias python="python 2.7"
```
