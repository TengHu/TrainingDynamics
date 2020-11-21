#!/bin/bash


FROM=$1

scp -r -o StrictHostKeyChecking=no -o ProxyCommand='/usr/local/bin/ussh bastion.uber.com -W %h:%p' -P 30701 "root@10.207.72.52:$FROM" "."

#scp -r -o StrictHostKeyChecking=no -o ProxyCommand='/usr/local/bin/ussh bastion.uber.com -W %h:%p' -P 31201 "./$FROM" "root@10.207.72.25:." 