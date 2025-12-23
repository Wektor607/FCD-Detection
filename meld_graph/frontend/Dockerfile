FROM node:20

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

# Запуск дев-сервера React
CMD ["npm", "start"]
