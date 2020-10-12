#!/bin/bash


FROM=$1

#scp -r -o StrictHostKeyChecking=no -o ProxyCommand='/usr/local/bin/ussh bastion.uber.com -W %h:%p' -P 32101 "root@10.207.72.12:$FROM" "."

scp -r -o StrictHostKeyChecking=no -o ProxyCommand='/usr/local/bin/ussh bastion.uber.com -W %h:%p' -P 32101 "./$FROM" "root@10.207.72.12:." 