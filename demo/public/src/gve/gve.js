angular.module('gveApp', [])
    .controller('MainCtrl', ['$scope', '$http', function ($scope, $http) {
        $scope.loading = true;

        $scope.load_classes = function () {
            $scope.classes = [];
            $http.get('/api/classes').then(function (response) {
                $scope.classes = response.data.message;
                $scope.retry();
            });
        };

        $scope.retry = function () {
            $scope.loading = true;
            $scope.explanation = "";
            $scope.images = [];
            $scope.correct_class_label = "";
            $scope.result = false;
            var class1 = $scope.classes[Math.floor(Math.random() * $scope.classes.length)];
            var index = $scope.classes.indexOf(class1);
            var classes = angular.copy($scope.classes);
            classes.splice(index, 1);
            var class2 = classes[Math.floor(Math.random() * classes.length)];
            $http.get('/api/counterfactual', {
                params: {
                    class_true: class1,
                    class_false: class2
                }
            }).then(function (response) {
                $scope.explanation = response.data.message.images[0].explanation;
                $scope.correct_class_label = response.data.message.images[0].class_label;
                $scope.images = shuffle(response.data.message.images);
                $scope.loading = false;
                $scope.answered = false;
                $scope.result = false;
            });
        };

        $scope.load_classes();

        function shuffle(array) {
            var currentIndex = array.length, temporaryValue, randomIndex;

            // While there remain elements to shuffle...
            while (0 !== currentIndex) {

                // Pick a remaining element...
                randomIndex = Math.floor(Math.random() * currentIndex);
                currentIndex -= 1;

                // And swap it with the current element.
                temporaryValue = array[currentIndex];
                array[currentIndex] = array[randomIndex];
                array[randomIndex] = temporaryValue;
            }

            return array;
        }

        $scope.select = function (image) {
            $scope.answered = true;
            $scope.result = $scope.correct_class_label == image.class_label;
        };
    }]);
