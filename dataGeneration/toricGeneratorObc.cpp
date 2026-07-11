#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

inline static const std::string roundDouble(const double input,
                                            const int decimal_places) {
    std::ostringstream str;
    str << std::fixed << std::setprecision(decimal_places);
    str << input;
    return str.str();
}

std::map<std::vector<int>, int> initVertices(int L, int D) {
    std::map<std::vector<int>, int> vertices;
    for (int i = 0; i < pow(L, D); ++i) {
        std::vector<int> indices(D, 0);
        int temp = i;
        for (int d = D - 1; d >= 0; --d) {
            indices[d] = temp % L;
            temp /= L;
        }
        vertices[indices] = 1;
    }
    return vertices;
}

std::map<std::tuple<std::vector<int>, int, int>, int>
initPlaquettes(int L, int D, std::mt19937 &gen, bool random = false,
               bool open = false) {
    std::map<std::tuple<std::vector<int>, int, int>, int> plaquettes;
    std::uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < pow(L, D); ++i) {
        std::vector<int> indices(D, 0);
        int temp = i;
        for (int d = D - 1; d >= 0; --d) {
            indices[d] = temp % L;
            temp /= L;
        }
        for (int d1 = 0; d1 < D; ++d1) {
            for (int d2 = d1 + 1; d2 < D; ++d2) {
                int value = 1;
                if (random && open && (indices[0] != L - 1)) {
                    value = (dis(gen) == 0) ? -1 : 1;
                }
                plaquettes[std::make_tuple(indices, d1, d2)] = value;
            }
        }
    }
    return plaquettes;
}

// Fixed: Passed by reference (&) so changes persist
void applyLogical(std::map<std::tuple<std::vector<int>, int>, int> &bonds,
                  int direction, int L, int D) {
    std::vector<int> vertex(D, L / 2);
    for (int l = 0; l < L; ++l) {
        vertex[direction] = (vertex[direction] + 1) % L;
        auto curBond = std::make_tuple(vertex, direction);
        bonds[curBond] *= -1;
    }
}

std::map<std::tuple<std::vector<int>, int>, int> populateBonds(
    int L, int D,
    const std::map<std::tuple<std::vector<int>, int, int>, int> &plaquettes) {
    std::map<std::tuple<std::vector<int>, int>, int> bonds;
    for (int i = 0; i < pow(L, D); ++i) {
        std::vector<int> indices(D, 0);
        int temp = i;
        for (int d = D - 1; d >= 0; --d) {
            indices[d] = temp % L;
            temp /= L;
        }
        for (int bDir = 0; bDir < D; ++bDir) {
            auto curBond = std::make_tuple(indices, bDir);
            int bVal = 1;
            for (int j = 0; j < D; ++j) {
                if (j != bDir) {
                    std::vector<int> altVindex = indices;
                    altVindex[j] = (altVindex[j] - 1 + L) % L;
                    auto faceTuple = std::make_tuple(indices, std::min(bDir, j),
                                                     std::max(bDir, j));
                    auto altFaceTuple = std::make_tuple(
                        altVindex, std::min(bDir, j), std::max(bDir, j));

                    // Use .at() or find() for const map safety
                    bVal *=
                        plaquettes.at(faceTuple) * plaquettes.at(altFaceTuple);
                }
            }
            bonds[curBond] = bVal;
        }
    }
    return bonds;
}

void applyError(
    int L, std::map<std::vector<int>, int> &vertices,
    std::map<std::tuple<std::vector<int>, int, int>, int> &plaquettes,
    std::map<std::tuple<std::vector<int>, int>, int> &bonds,
    const std::vector<int> &vindex, int bond, double p, std::mt19937 &gen,
    int D) {

    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(gen) < p) {
        // As per original logic: errorType is forced to 1
        int errorType = 1;

        std::vector<int> secondVertex = vindex;
        secondVertex[bond] = (secondVertex[bond] + 1) % L;
        bonds[std::make_tuple(vindex, bond)] *= -1;

        if (errorType == 1) {
            vertices[vindex] *= -1;
            vertices[secondVertex] *= -1;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " L Lsmall D numSamples baseSeed"
                  << std::endl;
        return 1;
    }

    int L = std::stoi(argv[1]);
    int Lsmall = std::stoi(argv[2]);
    int D = std::stoi(argv[3]);
    int numSamples = std::stoi(argv[4]);
    int baseSeed = std::stoi(argv[5]);

#pragma omp parallel for
    for (int pi = 0; pi < 20; pi++) {
        double p = 0. + pi * 0.01;

        std::mt19937 gen(baseSeed * 10000 + pi * 777);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        std::ostringstream filename;
        std::string type = (baseSeed == 0) ? "traindata" : "testdata";

        // Adjust path as needed
        filename << "/Users/tarun/phd/nqsWeight/" << type
                 << "/toricopenpqv3/d=" << D << "/measurements_L=" << Lsmall
                 << "_p=" << roundDouble(p, 3) << ".csv";
        std::ofstream fileObj(filename.str(), std::ios::out | std::ios::trunc);

        std::vector<std::string> buffer;
        const int bufferSize = 1000;

        for (int n = 0; n < numSamples; n++) {
            auto vertices = initVertices(L, D);
            auto plaquettes = initPlaquettes(L, D, gen, true, true);
            auto bonds = populateBonds(L, D, plaquettes);

            // if (dis(gen) < 0.5) {
            //     applyLogical(bonds, 1, L, D);
            // }

            for (auto const &[vindex, val] : vertices) {
                for (int i = 0; i < D; ++i) {
                    if (vindex[0] == (Lsmall - 1) && i == 0)
                        continue;
                    applyError(L, vertices, plaquettes, bonds, vindex, i, p,
                               gen, D);
                }
            }

            std::ostringstream line;
            if (D == 2) {
                for (int i = 0; i < D; ++i) {
                    for (int vi = 0; vi < Lsmall; vi++) {
                        for (int vj = 0; vj < Lsmall; vj++) {
                            std::vector<int> vertex = {vi, vj};
                            auto curBond = std::make_tuple(vertex, i);
                            int bondVal = bonds[curBond];

                            if ((vj == 0) && (i == 0)) {
                                bondVal = 1;
                            }
                            line << bondVal << " ";
                        }
                    }
                }
            } else if (D == 1) {
                for (int vi = 0; vi < Lsmall; vi++) {
                    std::vector<int> vertex = {vi};
                    line << vertices[vertex] << " ";
                }
            }

            buffer.push_back(line.str());
            if (buffer.size() >= bufferSize) {
                for (const auto &l : buffer)
                    fileObj << l << '\n';
                buffer.clear();
            }
        }
        for (const auto &l : buffer)
            fileObj << l << '\n';
        fileObj.close();
    }
    return 0;
}
